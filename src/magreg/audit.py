"""Leak audit: does a feature track amplitude once magnitude and distance are fixed?

Scale invariance (enforced by the tests) removes the absolute level, but a
shape measured on a noisy window still depends on how far the signal rises
above the noise: a low-SNR spectrum flattens, a low-SNR envelope never decays
to 0.2. Such a feature carries SNR — i.e. amplitude — into the model.

At fixed magnitude and distance the remaining SNR variation is station noise,
site gain and radiation pattern, none of which a clean source feature should
follow. So for each ``f_`` column we report the partial Spearman correlation
with ``ref_log10_snr_s`` (and ``_p``) given magnitude and log hypocentral
distance. Large values mean noise contamination, not source physics.

That cross-event version over-flags real source features: catalogue
magnitudes have errors, and an event truly larger than its ML has both
higher SNR and a lower corner. So the primary score is **within-event**:
ranks are demeaned per event (event fixed effects, removing the source
entirely) and residualised on log distance before correlating with SNR.
Inside one event the source is identical; SNR then varies only with
station noise, site and radiation, and a feature that follows it is
reading the noise floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return y - a @ coef


def partial_spearman(df: pd.DataFrame, a: str, b: str, controls: list[str]) -> float:
    d = df[[a, b, *controls]].dropna().rank()
    if len(d) < 30:
        return np.nan
    x = d[controls].to_numpy()
    ra, rb = _resid(d[a].to_numpy(), x), _resid(d[b].to_numpy(), x)
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def within_event_spearman(df: pd.DataFrame, a: str, b: str, control: str) -> float:
    """Partial rank correlation of a and b with event fixed effects + control."""
    d = df[["event_id", a, b, control]].dropna()
    d = d[d.groupby("event_id")["event_id"].transform("size") >= 3]
    if len(d) < 30:
        return np.nan
    r = d[[a, b, control]].rank()
    r = r - r.groupby(d["event_id"]).transform("mean")
    x = r[[control]].to_numpy()
    ra, rb = _resid(r[a].to_numpy(), x), _resid(r[b].to_numpy(), x)
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def leak_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(log_hypo=np.log10(df["ctl_hypo_km"]))
    ctl = ["mag", "log_hypo"]
    rows = []
    for c in sorted(col for col in df.columns if col.startswith("f_")):
        rows.append({
            "feature": c,
            "rho_mag": df[[c, "mag"]].corr(method="spearman").iloc[0, 1],
            "prho_mag|dist": partial_spearman(df, c, "mag", ["log_hypo"]),
            "prho_snr_s|mag,dist": partial_spearman(df, c, "ref_log10_snr_s", ctl),
            "prho_snr_p|mag,dist": partial_spearman(df, c, "ref_log10_snr_p", ctl),
            "within_snr_s|dist": within_event_spearman(df, c, "ref_log10_snr_s", "log_hypo"),
            "within_snr_p|dist": within_event_spearman(df, c, "ref_log10_snr_p", "log_hypo"),
        })
    t = pd.DataFrame(rows).set_index("feature")
    t["leak_score"] = t[["within_snr_s|dist", "within_snr_p|dist"]].abs().max(axis=1)
    t["leak_score_cross_event"] = t[["prho_snr_s|mag,dist", "prho_snr_p|mag,dist"]].abs().max(axis=1)
    return t.sort_values("leak_score", ascending=False)
