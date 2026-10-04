import numpy as np
import pandas as pd

from magreg.train import TrainConfig, family, make_folds, run


def fake(n_ev=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    stations = [f"S{i:02d}" for i in range(25)]
    for e in range(n_ev):
        m = 2 + rng.exponential(0.5)
        for s in rng.choice(stations, rng.integers(3, 8), replace=False):
            d = rng.uniform(10, 200)
            rows.append({"event_id": e, "station": s, "mag": m,
                         "f_sh_qc_brune_fc_hz": 10 ** (1.5 - 0.4 * m + rng.normal(0, 0.15)),
                         "f_pz_tau_c": rng.normal(), "f_noise": rng.normal(),
                         "ctl_hypo_km": d,
                         "ref_log10_peak_counts": m - 1.5 * np.log10(d) + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def test_both_split_is_doubly_disjoint():
    df = fake()
    for _, _, tr, te in make_folds(df, TrainConfig(folds=4, repeats=2)):
        assert not set(df.loc[tr, "event_id"]) & set(df.loc[te, "event_id"])
        assert not set(df.loc[tr, "station"]) & set(df.loc[te, "station"])


def test_run_learns_signal_and_ranks_it():
    cfg = TrainConfig(split="event", folds=3, num_boost_round=300, early_stopping=30)
    res = run(fake(), cfg, sets=("shape", "ctl", "amplitude"), log=lambda *_: None)
    s = res["summary"]["ev_mae_mean"]
    assert s["shape"] < s["ctl"]
    assert res["shap"].index[0] in ("f_sh_qc_brune_fc_hz", "f_evmed_sh_qc_brune_fc_hz")
    assert res["families"].index[0] in ("S_source_attcorr", "event_median")


def test_family_names():
    assert family("f_sh_qc_brune_fc_hz") == "S_source_attcorr"
    assert family("f_pz_brune_kappa") == "P_source"
    assert family("f_lpdt_slope_dec_per_s") == "P_lpdt"
    assert family("f_durh_d5_95_s") == "sig_duration"
    assert family("f_sh_T_dv_s") == "peak_periods"
    assert family("f_pz_slope") == "P_spectrum"
