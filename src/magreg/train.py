"""LightGBM magnitude regression with leak-aware baselines and importance.

Feature sets (all trained on identical splits, so their scores compare):

* ``shape``      ``f_`` + ``ctl_`` — the model this project is about
* ``shape_only`` ``f_`` alone — how much the shapes say without distance
* ``ctl``        ``ctl_`` alone — the distance/depth floor (selection effect)
* ``amplitude``  ``ref_log10_peak_counts`` + ``ctl_`` — a fitted local
                 magnitude relation; the "cheating" ceiling

The useful number is how much of the gap between ``ctl`` and ``amplitude``
the shape model closes.

Splits are grouped. ``event``: no event in both train and test. ``both``
(default): events *and* stations are partitioned into K groups and fold k
tests only on records whose event and station are both in group k, training
on records with neither — no shared event, no shared site response. Each
repeat re-draws the partition; the spread across folds × repeats is the
honest error bar.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_SETS = ("shape", "shape_only", "ctl", "amplitude")


@dataclass
class TrainConfig:
    split: str = "both"            # "both" | "event"
    folds: int = 5
    repeats: int = 3
    seed: int = 0
    weight: str = "event"          # "event": each event sums to 1; "none"
    tail_weight: bool = False      # also flatten the magnitude histogram
    event_medians: bool = True     # add f_evmed_* (shape sets only)
    val_frac: float = 0.15         # inner early-stopping split, by event
    params: dict = field(default_factory=lambda: {
        "objective": "huber", "alpha": 0.5,
        "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1, "num_threads": 0,
    })
    num_boost_round: int = 5000
    early_stopping: int = 200


# --------------------------------------------------------------------------
# Columns and feature families
# --------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame, which: str, drop: set[str] = frozenset()) -> list[str]:
    f = [c for c in df.columns if c.startswith("f_") and c not in drop]
    ctl = [c for c in df.columns if c.startswith("ctl_")]
    return {
        "shape": f + ctl,
        "shape_only": f,
        "ctl": ctl,
        "amplitude": ["ref_log10_peak_counts"] + ctl,
    }[which]


_SPECTRUM = re.compile(r"centroid_hz|bw_logf|flatness|entropy$|slope|peak_hz|rolloff|frac_")
_SOURCE = re.compile(r"brune|boat|fc_at_edge|snoke")
_STATS = re.compile(r"kurt|skew|crest|zcr|hjorth")
_CPLX = re.compile(r"perm_ent|samp_ent|higuchi|svd_ent")


def family(col: str) -> str:
    """Group a column into a physically meaningful feature family."""
    if col.startswith("ctl_"):
        return "control:" + col[4:]
    if col.startswith("ref_"):
        return "amplitude"
    if col.startswith("f_evmed_"):
        return "event_median"
    if col.startswith("f_lpdt_"):
        return "P_lpdt"
    if col.startswith(("f_durz_", "f_durh_")):
        return "sig_duration"
    if col.startswith("f_fc_p_over_s"):
        return "fc_P/S_ratio"
    if col.endswith(("T_dv_s", "T_va_s")):
        return "peak_periods"
    if col.startswith(("f_ppol_", "f_spol_")) or col == "f_log_s_over_p":
        return "polarisation+S/P"
    if col.startswith("f_envz_"):
        return "envelope_Z"
    if col.startswith("f_envh_"):
        return "envelope_H"
    if "tau_" in col:
        return "P_tau"
    win = "P" if col.startswith("f_pz_") else "S"
    if "_qc_" in col:
        return f"{win}_source_attcorr"
    for name, rx in (("source", _SOURCE), ("spectrum", _SPECTRUM),
                     ("stats", _STATS), ("complexity", _CPLX)):
        if rx.search(col):
            return f"{win}_{name}"
    return "other"


EVENT_MEDIAN_COLS = [
    "f_pz_qc_brune_fc_hz", "f_sh_qc_brune_fc_hz", "f_pz_qc_snoke_fc_hz", "f_sh_qc_snoke_fc_hz",
    "f_sh_qc_brune_kappa", "f_pz_tau_c", "f_pz_tau_p_max", "f_lpdt_t_plateau_s",
    "f_durh_d5_95_s", "f_sh_T_dv_s", "f_fc_p_over_s_snoke",
]


def with_event_medians(x: pd.DataFrame, event_ids: np.ndarray) -> pd.DataFrame:
    """Add ``f_evmed_*``: per-event median over the records in ``x``.

    Averages radiation pattern and site effects out of the source estimates.
    Called separately on each train and test subset so a fold only ever
    aggregates records it can see. Only medians, never the station count,
    which tracks magnitude through recording selection.
    """
    cols = [c for c in EVENT_MEDIAN_COLS if c in x.columns]
    if not cols:
        return x
    med = x[cols].groupby(event_ids).transform("median")
    med.columns = ["f_evmed_" + c[2:] for c in cols]
    return pd.concat([x, med], axis=1)


# --------------------------------------------------------------------------
# Splits and weights
# --------------------------------------------------------------------------


def make_folds(df: pd.DataFrame, cfg: TrainConfig):
    """Yield ``(repeat, fold, train_mask, test_mask)``."""
    events = df["event_id"].unique()
    stations = df["station"].unique()
    for r in range(cfg.repeats if cfg.split == "both" else 1):
        rng = np.random.default_rng(cfg.seed + r)
        eg = pd.Series(rng.permutation(len(events)) % cfg.folds, index=events)
        sg = pd.Series(rng.permutation(len(stations)) % cfg.folds, index=stations)
        e = df["event_id"].map(eg).to_numpy()
        s = df["station"].map(sg).to_numpy()
        for k in range(cfg.folds):
            if cfg.split == "event":
                yield r, k, e != k, e == k
            else:
                yield r, k, (e != k) & (s != k), (e == k) & (s == k)


def sample_weights(df: pd.DataFrame, cfg: TrainConfig) -> np.ndarray:
    w = np.ones(len(df))
    if cfg.weight == "event":
        w /= df.groupby("event_id")["event_id"].transform("size").to_numpy()
    if cfg.tail_weight:
        b = np.floor((df["mag"] - 2.0) / 0.2 + 1e-6)
        ev = df.drop_duplicates("event_id")
        nb = np.floor((ev["mag"] - 2.0) / 0.2 + 1e-6).value_counts()
        inv = (nb.max() / b.map(nb)).clip(upper=10.0).to_numpy()  # cap at 10x
        w *= inv
    return w / w.mean()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err**2) / ss) if ss > 0 else np.nan,
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
        "n": int(len(y)),
    }


def event_level(pred: pd.DataFrame) -> pd.DataFrame:
    """Median of station predictions per event (and per fold, if repeated)."""
    keys = [k for k in ("repeat", "fold") if k in pred] + ["event_id"]
    return pred.groupby(keys, as_index=False).agg(mag=("mag", "first"), pred=("pred", "median"),
                                                 n_sta=("pred", "size"))


def by_magnitude(ev: pd.DataFrame, width: float = 0.5) -> pd.DataFrame:
    b = np.floor(ev["mag"] / width) * width
    g = ev.assign(err=ev["pred"] - ev["mag"]).groupby(b)
    return pd.DataFrame({"n": g.size(), "mae": g["err"].apply(lambda e: e.abs().mean()),
                         "bias": g["err"].mean()}).rename_axis("mag_bin")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _fit(x, y, w, ev_ids, cfg: TrainConfig, seed: int) -> lgb.Booster:
    rng = np.random.default_rng(seed)
    uniq = np.unique(ev_ids)
    val_ev = set(rng.choice(uniq, max(1, int(cfg.val_frac * len(uniq))), replace=False))
    v = np.isin(ev_ids, list(val_ev))
    params = cfg.params | {"seed": seed}
    dtr = lgb.Dataset(x[~v], y[~v], weight=w[~v], free_raw_data=False)
    dva = lgb.Dataset(x[v], y[v], weight=w[v], reference=dtr)
    return lgb.train(params, dtr, cfg.num_boost_round, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False)])


def group_permutation(bst, x: pd.DataFrame, y_ev: pd.DataFrame, cols: list[str],
                      rng, base_mae: float) -> dict[str, float]:
    """Event-level MAE increase when a whole feature family is shuffled.

    Shuffling a family jointly (rows permuted together) avoids the
    correlated-feature trap where each member alone looks unimportant.
    """
    fams: dict[str, list[int]] = {}
    for i, c in enumerate(cols):
        fams.setdefault(family(c), []).append(i)
    xv = x.to_numpy()
    out = {}
    for fam, idx in fams.items():
        xp = xv.copy()
        perm = rng.permutation(len(xp))
        xp[:, idx] = xv[perm][:, idx]
        p = pd.Series(bst.predict(xp, num_iteration=bst.best_iteration), index=x.index)
        ev = y_ev.assign(pred=p).groupby("event_id").agg(mag=("mag", "first"), pred=("pred", "median"))
        out[fam] = float(np.mean(np.abs(ev["pred"] - ev["mag"])) - base_mae)
    return out


def run(df: pd.DataFrame, cfg: TrainConfig, sets=FEATURE_SETS, drop: set[str] = frozenset(),
        out: Path | None = None, log=print) -> dict:
    df = df.reset_index(drop=True)
    y = df["mag"].to_numpy()
    w = sample_weights(df, cfg)
    folds = list(make_folds(df, cfg))
    rng = np.random.default_rng(cfg.seed)
    const_rows = []
    for r, k, tr, te in folds:
        if te.sum() == 0:
            continue
        mu = np.average(y[tr], weights=w[tr])
        ev = df.loc[te].drop_duplicates("event_id")["mag"].to_numpy()
        const_rows.append(float(np.mean(np.abs(ev - mu))))
    results, preds, shap_rows, perm_rows = {}, [], [], []
    for name in sets:
        cols = feature_columns(df, name, drop)
        fold_rows = []
        for r, k, tr, te in folds:
            if te.sum() == 0:
                continue
            xtr, xte = df.loc[tr, cols], df.loc[te, cols]
            if cfg.event_medians and name.startswith("shape"):
                xtr = with_event_medians(xtr, df.loc[tr, "event_id"].to_numpy())
                xte = with_event_medians(xte, df.loc[te, "event_id"].to_numpy())
            fcols = list(xtr.columns)
            bst = _fit(xtr.to_numpy(), y[tr], w[tr], df.loc[tr, "event_id"].to_numpy(),
                       cfg, cfg.seed + 100 * r + k)
            p = bst.predict(xte.to_numpy(), num_iteration=bst.best_iteration)
            pr = df.loc[te, ["event_id", "station", "mag"]].assign(pred=p, repeat=r, fold=k, set=name)
            preds.append(pr)
            ev = event_level(pr)
            m_rec, m_ev = metrics(pr["mag"].to_numpy(), p), metrics(ev["mag"].to_numpy(), ev["pred"].to_numpy())
            fold_rows.append({"repeat": r, "fold": k, "trees": bst.best_iteration,
                              **{f"rec_{a}": b for a, b in m_rec.items()},
                              **{f"ev_{a}": b for a, b in m_ev.items()}})
            if name == "shape":
                contrib = bst.predict(xte.to_numpy(), num_iteration=bst.best_iteration, pred_contrib=True)
                shap_rows.append(pd.Series(np.abs(contrib[:, :-1]).mean(0), index=fcols))
                perm = group_permutation(bst, xte, df.loc[te, ["event_id", "mag"]], fcols, rng,
                                         m_ev["mae"])
                perm_rows.append(perm)
            log(f"  {name:<11} r{r} f{k}: event MAE {m_ev['mae']:.3f} "
                f"(n_ev={m_ev['n']}, trees={bst.best_iteration})")
        fr = pd.DataFrame(fold_rows)
        results[name] = {
            "folds": fr.to_dict("records"),
            "ev_mae_mean": float(fr["ev_mae"].mean()), "ev_mae_sd": float(fr["ev_mae"].std()),
            "rec_mae_mean": float(fr["rec_mae"].mean()), "rec_mae_sd": float(fr["rec_mae"].std()),
            "ev_r2_mean": float(fr["ev_r2"].mean()), "ev_spearman_mean": float(fr["ev_spearman"].mean()),
            "n_features": len(fcols),
        }

    pred = pd.concat(preds, ignore_index=True)
    shap = pd.concat(shap_rows, axis=1)
    shap_tbl = pd.DataFrame({"mean_abs_shap": shap.mean(1), "sd": shap.std(1)})
    shap_tbl["family"] = [family(c) for c in shap_tbl.index]
    shap_tbl = shap_tbl.sort_values("mean_abs_shap", ascending=False)
    fam_shap = shap_tbl.groupby("family")["mean_abs_shap"].sum()
    perm = pd.DataFrame(perm_rows)
    fam_tbl = pd.DataFrame({"perm_delta_mae": perm.mean(), "perm_sd": perm.std(),
                            "sum_abs_shap": fam_shap}).sort_values("perm_delta_mae", ascending=False)
    ev_shape = event_level(pred[pred["set"] == "shape"])
    bymag = by_magnitude(ev_shape)

    summary = pd.DataFrame({k: {m: v[m] for m in ("ev_mae_mean", "ev_mae_sd", "rec_mae_mean",
                                                   "ev_r2_mean", "ev_spearman_mean", "n_features")}
                            for k, v in results.items()}).T
    if {"ctl", "amplitude", "shape"} <= set(results):
        lo, hi = results["ctl"]["ev_mae_mean"], results["amplitude"]["ev_mae_mean"]
        gap = (lo - results["shape"]["ev_mae_mean"]) / (lo - hi) if lo != hi else np.nan
    else:
        gap = np.nan
    const = float(np.mean(const_rows))  # train-fold weighted mean, scored per test event

    if out:
        out.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(out / "predictions.parquet", index=False)
        shap_tbl.to_csv(out / "importance_shap.csv")
        fam_tbl.to_csv(out / "importance_families.csv")
        bymag.to_csv(out / "by_magnitude.csv")
        summary.to_csv(out / "summary.csv")
        (out / "metrics.json").write_text(json.dumps(
            {"config": asdict(cfg), "dropped_features": sorted(drop), "gap_closed": gap,
             "constant_mae_event": const, "sets": results}, indent=2, default=float))
    return {"summary": summary, "gap_closed": gap, "constant_mae": const,
            "shap": shap_tbl, "families": fam_tbl, "by_mag": bymag}
