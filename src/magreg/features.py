"""Amplitude-invariant waveform features.

Every function here returns the same value when its input is multiplied by
any non-zero constant: windows are normalised by their own scale, and only
ratios, times, shapes and orderings are reported. Nothing is measured
relative to the pre-event noise, because the noise level is roughly fixed and
any signal/noise quantity is an amplitude meter in disguise.
``tests/test_invariance.py`` enforces this for the whole record pipeline.
"""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
from scipy.signal import hilbert
from scipy.signal.windows import dpss
from scipy.spatial.distance import pdist
from scipy.stats import kurtosis, skew

EPS = 1e-12


def unit(x: np.ndarray) -> np.ndarray:
    """Demeaned ``x`` scaled to unit peak. All features start from this."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    s = np.max(np.abs(x))
    return x / s if s > 0 else x


# --------------------------------------------------------------------------
# Spectral shape
# --------------------------------------------------------------------------

SPEC_BANDS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 45)]


def mt_power(x: np.ndarray, fs: float, nw: float = 2.5, k: int = 4):
    """Multitaper power spectrum of one window (or the mean over rows)."""
    x = np.atleast_2d(x)
    n = x.shape[1]
    tapers = dpss(n, nw, k)
    p = np.zeros(n // 2 + 1)
    for row in x:
        row = unit(row)
        p += (np.abs(np.fft.rfft(tapers * row, axis=1)) ** 2).mean(0)
    return np.fft.rfftfreq(n, 1.0 / fs), p / len(x)


def spectral_shape(f, p, fmin: float, fmax: float) -> dict:
    """Shape descriptors of a velocity power spectrum on [fmin, fmax]."""
    m = (f >= fmin) & (f <= fmax)
    f, p = f[m], p[m]
    w = p / max(p.sum(), EPS)
    lf = np.log(f)
    centroid = float(np.sum(w * lf))
    cdf = np.cumsum(w)
    out = {
        "centroid_hz": math.exp(centroid),
        "bw_logf": math.sqrt(max(np.sum(w * (lf - centroid) ** 2), 0.0)),
        "flatness": float(np.exp(np.mean(np.log(p + EPS))) / max(p.mean(), EPS)),
        "entropy": float(-np.sum(w * np.log(w + EPS)) / math.log(len(w))),
        "slope": float(np.polyfit(lf, np.log(p + EPS), 1)[0]),
        "peak_hz": float(f[np.argmax(p)]),
    }
    for q in (0.5, 0.85, 0.95):
        out[f"rolloff{int(q * 100)}_hz"] = float(f[min(np.searchsorted(cdf, q), len(f) - 1)])
    tot = max(p.sum(), EPS)
    for lo, hi in SPEC_BANDS:
        if hi > fmin and lo < fmax:
            sel = (f >= lo) & (f < hi)
            out[f"frac_{lo}_{hi}"] = float(p[sel].sum() / tot)
    return out


FC_GRID = np.geomspace(0.3, 45.0, 80)
N_GRID = np.linspace(1.0, 4.0, 31)


def source_fit(f, p, fmin: float, fmax: float, npts: int = 40) -> dict:
    """Fit source models to the displacement amplitude spectrum.

    * Brune with attenuation: ``log A = log O + log 1/(1+(f/fc)^2) - pi*kappa*f``
    * Boatwright, free falloff: ``log A = log O - log(1+(f/fc)^n)``

    ``O`` absorbs the absolute level, so only shape (fc, kappa, n) remains.
    Spectra are resampled on log-spaced frequencies so the fit is not
    dominated by the many high-frequency bins.
    """
    m = (f >= fmin) & (f <= fmax) & (f > 0)
    disp = np.sqrt(p[m]) / (2 * np.pi * f[m])
    fl = np.geomspace(max(fmin, f[m][0]), min(fmax, f[m][-1]), npts)
    y = np.interp(np.log(fl), np.log(f[m]), np.log(disp + EPS))
    ones = np.ones_like(fl)

    best = (np.inf, np.nan, np.nan)
    for fc in FC_GRID:
        g = y + np.log1p((fl / fc) ** 2)
        a = np.column_stack([ones, -np.pi * fl])
        coef, *_ = np.linalg.lstsq(a, g, rcond=None)
        r = float(np.mean((g - a @ coef) ** 2))
        if r < best[0]:
            best = (r, fc, coef[1])
    out = {"brune_fc_hz": best[1], "brune_kappa": best[2], "brune_rms": math.sqrt(best[0])}

    best = (np.inf, np.nan, np.nan)
    for n in N_GRID:
        for fc in FC_GRID:
            g = y + np.log1p((fl / fc) ** n)
            r = float(np.var(g))  # least-squares O is the mean
            if r < best[0]:
                best = (r, fc, n)
    out.update({"boat_fc_hz": best[1], "boat_n": best[2], "boat_rms": math.sqrt(best[0])})
    out["fc_at_edge"] = float(
        min(out["brune_fc_hz"], out["boat_fc_hz"]) <= FC_GRID[0] * 1.001
        or max(out["brune_fc_hz"], out["boat_fc_hz"]) >= FC_GRID[-1] * 0.999
    )
    return out


# --------------------------------------------------------------------------
# Early-warning period parameters
# --------------------------------------------------------------------------


def tau_c(v: np.ndarray, fs: float) -> float:
    """Kanamori's tau_c on a high-passed P velocity window."""
    v = unit(v)
    u = np.cumsum(v) / fs
    u -= np.linspace(u[0], u[-1], len(u))  # remove integration drift
    return float(2 * np.pi * math.sqrt(np.sum(u**2) / max(np.sum(v**2), EPS)))


def tau_p_max(v: np.ndarray, fs: float, alpha: float = 0.99, skip_s: float = 0.5) -> float:
    """Allen & Kanamori's recursive predominant period, max after ``skip_s``."""
    v = unit(v)
    dv = np.gradient(v) * fs
    x = d = 0.0
    tp = np.empty(len(v))
    for i in range(len(v)):
        x = alpha * x + v[i] ** 2
        d = alpha * d + dv[i] ** 2
        tp[i] = 2 * np.pi * math.sqrt(x / d) if d > 0 else 0.0
    return float(tp[int(skip_s * fs):].max())


# --------------------------------------------------------------------------
# Waveform statistics and complexity
# --------------------------------------------------------------------------


def waveform_stats(x: np.ndarray) -> dict:
    x = unit(x)
    rms = math.sqrt(max(np.mean(x**2), EPS))
    dx, ddx = np.diff(x), np.diff(x, 2)
    mob = math.sqrt(np.var(dx) / max(np.var(x), EPS))
    mob_d = math.sqrt(np.var(ddx) / max(np.var(dx), EPS))
    return {
        "kurt": float(kurtosis(x)),
        "abs_skew": float(abs(skew(x))),  # sign is polarity, not size
        "crest": float(1.0 / rms),
        "zcr": float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))),
        "hjorth_mob": mob,
        "hjorth_cplx": mob_d / max(mob, EPS),
    }


def permutation_entropy(x: np.ndarray, order: int = 5, delay: int = 1) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = len(x) - (order - 1) * delay
    idx = np.arange(order)[None, :] * delay + np.arange(n)[:, None]
    ranks = np.argsort(x[idx], axis=1)
    codes = ranks @ (order ** np.arange(order))
    _, counts = np.unique(codes, return_counts=True)
    pr = counts / counts.sum()
    return float(-np.sum(pr * np.log(pr)) / math.log(math.factorial(order)))


def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """SampEn with tolerance ``r`` times the window's own std."""
    x = unit(x)
    L = len(x) - m
    if L < 2:
        return np.nan
    tol = r * np.std(x, ddof=1)
    idx = np.arange(m + 1)[None, :] + np.arange(L)[:, None]
    t = x[idx]
    b = np.count_nonzero(pdist(t[:, :m], metric="chebyshev") <= tol)
    a = np.count_nonzero(pdist(t, metric="chebyshev") <= tol)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    x = unit(x)
    n = len(x)
    ks = np.arange(1, kmax + 1)
    lk = []
    for k in ks:
        lm = []
        for m0 in range(k):
            s = x[m0::k]
            if len(s) < 2:
                continue
            lm.append(np.sum(np.abs(np.diff(s))) * (n - 1) / ((len(s) - 1) * k * k))
        lk.append(np.mean(lm))
    return float(-np.polyfit(np.log(ks), np.log(np.asarray(lk) + EPS), 1)[0])


def svd_entropy(x: np.ndarray, order: int = 10, delay: int = 1) -> float:
    x = unit(x)
    n = len(x) - (order - 1) * delay
    emb = x[np.arange(order)[None, :] * delay + np.arange(n)[:, None]]
    s = np.linalg.svd(emb, compute_uv=False)
    s = s / max(s.sum(), EPS)
    return float(-np.sum(s * np.log(s + EPS)) / math.log(order))


def complexity(x: np.ndarray) -> dict:
    return {
        "perm_ent": permutation_entropy(x),
        "samp_ent": sample_entropy(x),
        "higuchi_fd": higuchi_fd(x),
        "svd_ent": svd_entropy(x),
    }


# --------------------------------------------------------------------------
# Three-component polarisation
# --------------------------------------------------------------------------


def polarisation(z, n, e) -> dict:
    """Covariance-eigen polarisation; all three components share one scale."""
    m = np.vstack([z, n, e]).astype(np.float64)
    m -= m.mean(axis=1, keepdims=True)
    m /= max(np.max(np.abs(m)), EPS)
    w, v = np.linalg.eigh(np.cov(m))
    l3, l2, l1 = np.maximum(w, 0.0)
    v1 = v[:, 2]
    hz = math.sqrt(np.mean(n**2) + np.mean(e**2))
    return {
        "rectilin": float(1 - (l2 + l3) / (2 * max(l1, EPS))),
        "planarity": float(1 - 2 * l3 / max(l1 + l2, EPS)),
        "incidence_deg": float(np.degrees(np.arccos(min(abs(v1[0]), 1.0)))),
        "log_h_over_v": float(np.log10(max(hz, EPS) / max(math.sqrt(np.mean(z**2)), EPS))),
    }


# --------------------------------------------------------------------------
# Envelope shape
# --------------------------------------------------------------------------


def envelope(x: np.ndarray, fs: float, smooth_s: float = 0.2) -> np.ndarray:
    env = np.abs(hilbert(np.asarray(x, dtype=np.float64)))
    k = max(1, int(smooth_s * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    return env / max(env.max(), EPS)


def envelope_shape(env: np.ndarray, fs: float, i_p: int, i_s: int, t_origin_s: float) -> dict:
    """Shape of a peak-normalised envelope.

    ``i_p``/``i_s`` are the P and S sample indices inside ``env``;
    ``t_origin_s`` is the time of sample 0 after the origin. Decay is fitted
    from the peak to the end of the (fixed, S-anchored) window only.
    """
    sp = max((i_s - i_p) / fs, 1.0 / fs)
    ipk = int(np.argmax(env))
    after = env[i_p:]
    i10 = i_p + int(np.argmax(after >= 0.1))
    i90 = i_p + int(np.argmax(after >= 0.9))
    t = np.arange(len(env)) / fs
    tail = slice(ipk, len(env))
    lt = np.log(env[tail] + EPS)
    decay = np.polyfit(t[tail], lt, 1)[0] if len(lt) > 10 else np.nan
    tt = t[tail] + t_origin_s
    pexp = np.polyfit(np.log(np.maximum(tt, 1e-3)), lt, 1)[0] if len(lt) > 10 else np.nan
    w = env[i_p:]
    tc = np.sum(w * np.arange(len(w))) / max(w.sum(), EPS) / fs
    return {
        "t_peak_s": (ipk - i_p) / fs,
        "t_peak_rel_sp": (ipk - i_p) / fs / sp,
        "rise_10_90_s": (i90 - i10) / fs,
        "dur_above50_s": float(np.count_nonzero(env >= 0.5)) / fs,
        "dur_above20_s": float(np.count_nonzero(env >= 0.2)) / fs,
        "p_over_peak": float(env[i_p:i_s].max()) if i_s > i_p else np.nan,
        "decay_per_s": float(decay),
        "decay_powerlaw": float(pexp),
        "centroid_s": float(tc),
        "fill": float(w.mean()),
    }


# --------------------------------------------------------------------------
# Physics-based features
# --------------------------------------------------------------------------
# Each is still scale invariant: either a ratio of spectral integrals, a time
# read off a normalised curve, or a fit whose level parameter is discarded.


def attenuation_correct(f, p, travel_s: float, q0: float, qn: float):
    """Undo path attenuation exp(-pi f t / Q(f)) on a power spectrum.

    ``Q(f) = q0 * f**qn``. Anelastic loss over the travel time otherwise
    mimics a lower corner frequency — at M2–3 the two are confounded.
    Geometric spreading is frequency independent and cancels in shape.
    """
    q = q0 * np.maximum(f, 1e-3) ** qn
    return p * np.exp(2 * np.pi * f * travel_s / q)


def snoke_fc(f, p, fmin: float, fmax: float) -> float:
    """Non-parametric corner frequency, fc = sqrt(∫V² / ∫D²) / 2π.

    Andrews (1986) / Snoke (1987): the ratio of velocity- to displacement-
    spectrum energy. No source-model shape is assumed.
    """
    m = (f >= fmin) & (f <= fmax) & (f > 0)
    v2 = p[m]
    d2 = v2 / (2 * np.pi * f[m]) ** 2
    return float(math.sqrt(np.trapezoid(v2, f[m]) / max(np.trapezoid(d2, f[m]), EPS)) / (2 * np.pi))


def lpdt(v: np.ndarray, fs: float, skip_s: float = 0.05) -> dict:
    """Shape of the log P-displacement-vs-time curve (Colombelli & Zollo 2015).

    ``log10(running max |u|)`` relative to its final value grows and then
    plateaus; the plateau time tracks rupture half-duration. A ramp+plateau
    ``y = a + b*min(t, T)`` is fitted by grid search over T.
    """
    v = unit(v)
    u = np.cumsum(v) / fs
    u -= np.linspace(u[0], u[-1], len(u))
    pd_ = np.maximum.accumulate(np.abs(u))
    i0 = int(skip_s * fs)
    y = np.log10(np.maximum(pd_[i0:], EPS) / max(pd_[-1], EPS))
    t = np.arange(len(y)) / fs
    best = (np.inf, np.nan, np.nan)
    for tb in t[2:-2:2]:
        x = np.minimum(t, tb)
        a = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        r = float(np.sum((y - a @ coef) ** 2))
        if r < best[0]:
            best = (r, tb, coef[1])
    t90 = t[int(np.argmax(y >= math.log10(0.9)))]
    return {"t_plateau_s": float(best[1] + skip_s), "slope_dec_per_s": float(best[2]),
            "t90_s": float(t90 + skip_s), "fit_rms": math.sqrt(best[0] / len(y))}


def significant_duration(x: np.ndarray, fs: float) -> dict:
    """Husid-curve durations (Trifunac & Brady 1975): D5-75, D5-95, D20-80.

    ``x`` may be 2-D (components summed). Normalised cumulative energy, so the
    level cancels.
    """
    e = np.sum(np.atleast_2d(x).astype(np.float64) ** 2, axis=0)
    h = np.cumsum(e)
    h /= max(h[-1], EPS)

    def t(q):
        return float(np.searchsorted(h, q)) / fs

    return {"d5_75_s": t(0.75) - t(0.05), "d5_95_s": t(0.95) - t(0.05),
            "d20_80_s": t(0.80) - t(0.20), "t5_s": t(0.05)}


def peak_periods(v: np.ndarray, fs: float) -> dict:
    """2π·PGD/PGV and 2π·PGV/PGA from one high-passed velocity window.

    ``v`` may be 2-D (horizontal vector peaks). Ratios of peaks of the same
    record: period proxies used in early warning, level-free.
    """
    v = np.atleast_2d(v).astype(np.float64)
    v = v / max(np.max(np.abs(v)), EPS)
    d = np.cumsum(v, axis=1) / fs
    d -= np.linspace(d[:, :1], d[:, -1:], v.shape[1], axis=1)[..., 0]
    a = np.gradient(v, axis=1) * fs
    pk = [float(np.max(np.sqrt(np.sum(z**2, axis=0)))) for z in (d, v, a)]
    return {"T_dv_s": 2 * np.pi * pk[0] / max(pk[1], EPS),
            "T_va_s": 2 * np.pi * pk[1] / max(pk[2], EPS)}
