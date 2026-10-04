import numpy as np
import pandas as pd

from magreg.train import TrainConfig, concept_groups, family, make_folds, quantity, run


def fake(n_ev=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    stations = [f"S{i:02d}" for i in range(25)]
    for e in range(n_ev):
        m = 2 + rng.exponential(0.5)
        for s in rng.choice(stations, rng.integers(3, 8), replace=False):
            d = rng.uniform(10, 200)
            snr = m - 1.5 * np.log10(d) + rng.normal(0, 0.4)
            rows.append({"event_id": e, "station": s, "mag": m,
                         "ref_log10_snr_s": snr, "ref_log10_snr_p": snr + rng.normal(0, 0.1),
                         "f_leaky": snr + rng.normal(0, 0.05),  # reads the noise floor
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
    cfg.params["num_threads"] = 2
    res = run(fake(), cfg, sets=("shape", "ctl", "amplitude"), log=lambda *_: None)
    assert "f_leaky" not in res["shap"].index
    assert res["quantities"].index[0] == "f_sh_qc_brune_fc_hz"
    assert "f_evmed_sh_qc_brune_fc_hz" not in res["quantities"].index
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


def test_leaky_features_only_with_flag(tmp_path):
    import json

    cfg = TrainConfig(split="event", folds=2, num_boost_round=50, early_stopping=10)
    cfg.params["num_threads"] = 2
    df = fake(150)
    run(df, cfg, sets=("shape",), out=tmp_path / "a", log=lambda *_: None)
    m = json.loads((tmp_path / "a" / "metrics.json").read_text())
    assert m["include_leaky"] is False and "f_leaky" in m["dropped_features"]
    assert "f_sh_qc_brune_fc_hz" not in m["dropped_features"]

    res = run(df, cfg, sets=("shape",), out=tmp_path / "b", include_leaky=True, log=lambda *_: None)
    assert "f_leaky" in res["shap"].index
    assert json.loads((tmp_path / "b" / "metrics.json").read_text())["include_leaky"] is True


def test_quantity_merges_event_median():
    assert quantity("f_evmed_pz_qc_snoke_fc_hz") == "f_pz_qc_snoke_fc_hz"
    assert quantity("f_pz_qc_snoke_fc_hz") == "f_pz_qc_snoke_fc_hz"
    assert quantity("ctl_epi_km") == "ctl_epi_km"


def test_concepts_group_substitutes():
    cols = ["f_pz_qc_snoke_fc_hz", "f_evmed_sh_qc_snoke_fc_hz", "f_sh_brune_fc_hz",
            "f_fc_p_over_s_snoke", "ctl_epi_km", "ctl_hypo_km", "f_lpdt_t90_s", "f_pz_kurt"]
    g = concept_groups(cols)
    assert g["corner_freq_snoke_P+S"] == [0, 1]
    assert g["corner_freq_all"] == [0, 1, 2, 3]
    assert g["distance_all"] == [4, 5]
    assert g["rupture_duration_lpdt"] == [6]
