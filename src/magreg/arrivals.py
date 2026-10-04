"""Theoretical P/S travel times and an AIC refinement of the P pick.

TauP is far too slow to call per record, so the first P and S arrivals of
iasp91 are tabulated once on a (depth, distance) grid, cached to disk and
interpolated afterwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DEPTHS_KM = np.arange(0.0, 200.0 + 1e-9, 2.5)
DIST_DEG = np.arange(0.0, 12.0 + 1e-9, 0.05)
P_PHASES = ["p", "P", "Pn", "Pg"]
S_PHASES = ["s", "S", "Sn", "Sg"]


def _build_table(model: str) -> tuple[np.ndarray, np.ndarray]:
    from obspy.taup import TauPyModel

    taup = TauPyModel(model)
    tp = np.full((len(DEPTHS_KM), len(DIST_DEG)), np.nan)
    ts = np.full_like(tp, np.nan)
    for i, z in enumerate(DEPTHS_KM):
        for j, d in enumerate(DIST_DEG):
            arr = taup.get_travel_times(z, d, phase_list=P_PHASES + S_PHASES)
            p = [a.time for a in arr if a.name in P_PHASES]
            s = [a.time for a in arr if a.name in S_PHASES]
            tp[i, j] = min(p) if p else np.nan
            ts[i, j] = min(s) if s else np.nan
    return tp, ts


class TravelTimes:
    """First-arrival P and S times (s after origin) from an iasp91 table."""

    def __init__(self, cache: Path, model: str = "iasp91"):
        cache = Path(cache)
        if cache.exists():
            z = np.load(cache)
            tp, ts = z["tp"], z["ts"]
        else:
            tp, ts = _build_table(model)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, tp=tp, ts=ts)
        self._p = RegularGridInterpolator((DEPTHS_KM, DIST_DEG), tp)
        self._s = RegularGridInterpolator((DEPTHS_KM, DIST_DEG), ts)

    def __call__(self, depth_km: float, dist_km: float) -> tuple[float, float]:
        z = float(np.clip(depth_km, DEPTHS_KM[0], DEPTHS_KM[-1]))
        d = float(np.clip(dist_km / 111.195, DIST_DEG[0], DIST_DEG[-1]))
        return float(self._p((z, d))), float(self._s((z, d)))


def aic_pick(x: np.ndarray) -> int:
    """Maeda's AIC onset index of ``x``; invariant to the scale of ``x``."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 20:
        return n // 2
    k = np.arange(1, n - 1)
    c1 = np.cumsum(x**2)
    c2 = np.cumsum(x[::-1] ** 2)[::-1]
    # Variance of x[:k] and x[k:], computed from running moments.
    s1 = np.cumsum(x)
    s2 = np.cumsum(x[::-1])[::-1]
    var1 = c1[k - 1] / k - (s1[k - 1] / k) ** 2
    var2 = c2[k] / (n - k) - (s2[k] / (n - k)) ** 2
    tiny = np.finfo(float).tiny
    aic = k * np.log(np.maximum(var1, tiny)) + (n - k - 1) * np.log(np.maximum(var2, tiny))
    # Ignore the edges, where one side has too few samples to estimate.
    edge = max(5, n // 20)
    aic[:edge] = np.inf
    aic[-edge:] = np.inf
    return int(k[np.argmin(aic)])
