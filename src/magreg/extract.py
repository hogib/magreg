"""Per-record feature extraction: event mseed -> one row per station.

Column prefixes in the output table:

* ``f_``    amplitude-invariant features — the only model inputs
* ``ctl_``  path controls (distance, depth, azimuth); give these to the model
            so distance effects are not soaked up by the shape features
* ``ref_``  deliberately leaky references (absolute amplitude, SNR) for
            baselines and the leak audit — never model inputs
* everything else is metadata / label (``mag``, ``mag_type``, ...)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from . import features as F
from .arrivals import TravelTimes, aic_pick
from .catalog import back_azimuth, epicentral_km

CLIP_COUNTS = 8.0e6  # 24-bit digitisers saturate at 2**23 = 8.39e6


@dataclass(frozen=True)
class Config:
    noise: tuple[float, float] = (-6.0, -1.0)  # s relative to P, QC only
    p_len: float = 2.0                          # P window [P - 0.05, P + p_len - 0.05)
    s_pre: float = 0.2
    s_len: float = 5.0                          # S window [S - s_pre, S - s_pre + s_len)
    pol_len: float = 1.0                        # polarisation uses the first second
    env_pre: float = 1.0                        # envelope window [P - env_pre, S + env_post)
    env_post: float = 10.0
    pick_before: float = 1.5                    # AIC search [P_theo - before, P_theo + after];
    pick_after: float = 3.0                     # asymmetric: AFAD origins are whole seconds, so P is late
    bp: tuple[float, float] = (1.0, 40.0)
    hp: float = 0.5
    p_band: tuple[float, float] = (1.0, 40.0)
    s_band: tuple[float, float] = (0.7, 40.0)
    min_snr: float = 3.0                        # QC gate on P and S; never a feature
    q0: float = 60.0                            # S-wave Q(f) = q0 * f**qn; regional Turkish
    qn: float = 0.85                            #   estimates span ~40 f^1.0 to ~70 f^0.8
    qp_over_qs: float = 1.5
    lpdt_len: float = 3.0                       # LPDT curve, capped at S - 0.2 s


def _sos(fs: float, cfg: Config):
    return (
        butter(4, cfg.bp, btype="bandpass", fs=fs, output="sos"),
        butter(4, cfg.hp, btype="highpass", fs=fs, output="sos"),
    )


def _prep(x: np.ndarray, sos) -> np.ndarray:
    """Demean, detrend, 5% cosine taper, causal filter."""
    x = x - x.mean()
    t = np.arange(len(x))
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    n = max(1, int(0.05 * len(x)))
    w = np.ones(len(x))
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    w[:n], w[-n:] = ramp, ramp[::-1]
    return sosfilt(sos, x * w)


def _rms(x) -> float:
    return math.sqrt(max(float(np.mean(np.square(x))), 1e-300))


def _put(row: dict, prefix: str, d: dict):
    for k, v in d.items():
        row[f"{prefix}{k}"] = float(v)


def record_features(z, n, e, fs: float, start_s: float, tp_theo: float, ts_theo: float,
                    cfg: Config = Config()) -> tuple[dict | None, str]:
    """Features of one 3-component record.

    ``start_s`` is the time of sample 0 after the origin; ``tp_theo`` /
    ``ts_theo`` are theoretical arrival times after the origin. Returns
    ``(row, "ok")`` or ``(None, reason)``.
    """
    raw = np.vstack([z, n, e]).astype(np.float64)
    nsamp = raw.shape[1]
    sos_bp, sos_hp = _sos(fs, cfg)

    def idx(t_after_origin: float) -> int:
        return int(round((t_after_origin - start_s) * fs))

    sp_theo = ts_theo - tp_theo
    lo = idx(tp_theo - cfg.pick_before + cfg.noise[0] - 1.0)
    hi = idx(tp_theo + cfg.pick_after + sp_theo + max(cfg.env_post, cfg.s_len) + 1.0)
    if lo < 0 or hi > nsamp:
        return None, "window_outside_record"
    if not np.all(np.isfinite(raw[:, lo:hi])):
        return None, "gap"

    seg = raw[:, lo:hi]
    t0 = start_s + lo / fs  # time of seg[0] after origin
    bp = np.vstack([_prep(c, sos_bp) for c in seg])
    hp = np.vstack([_prep(c, sos_hp) for c in seg])

    def j(t: float) -> int:
        return int(round((t - t0) * fs))

    # P pick: AIC on band-passed Z, kept clear of the theoretical S.
    a, b = j(tp_theo - cfg.pick_before), j(min(tp_theo + cfg.pick_after, ts_theo - 0.3))
    if b - a < 50:
        return None, "pick_window_too_short"
    ip = a + aic_pick(bp[0, a:b])
    tp = t0 + ip / fs
    ts = tp + sp_theo
    i_s = j(ts)

    p0, p1 = ip - int(0.05 * fs), ip - int(0.05 * fs) + int(cfg.p_len * fs)
    s0 = i_s - int(cfg.s_pre * fs)
    s1 = s0 + int(cfg.s_len * fs)
    n0, n1 = ip + int(cfg.noise[0] * fs), ip + int(cfg.noise[1] * fs)
    e0, e1 = ip - int(cfg.env_pre * fs), i_s + int(cfg.env_post * fs)
    if n0 < 0 or max(s1, e1) > bp.shape[1]:
        return None, "window_outside_record"

    # ---- QC and leaky references (never features) -------------------------
    snr_p = _rms(bp[0, p0:p1]) / _rms(bp[0, n0:n1])
    snr_s = _rms(bp[1:, s0:s1]) / _rms(bp[1:, n0:n1])
    span = seg[:, p0:e1] - np.median(seg[:, n0:n1], axis=1, keepdims=True)
    peak = np.max(np.abs(span), axis=1)
    at_peak = int(max(np.count_nonzero(np.abs(span[c]) >= 0.999 * peak[c]) for c in range(3)))
    clipped = bool(np.max(np.abs(seg[:, p0:e1])) >= CLIP_COUNTS or at_peak >= 5)
    vec = np.sqrt(np.sum(bp[:, p0:e1] ** 2, axis=0))
    row: dict = {
        "ref_log10_peak_counts": math.log10(max(vec.max(), 1e-300)),
        "ref_log10_snr_p": math.log10(snr_p),
        "ref_log10_snr_s": math.log10(snr_s),
        "ref_clipped": float(clipped),
        "pick_shift_s": tp - tp_theo,
    }
    if clipped:
        return None, "clipped"
    if snr_p < cfg.min_snr:
        return None, "low_snr_p"
    if snr_s < cfg.min_snr:
        return None, "low_snr_s"

    # ---- features -----------------------------------------------------------
    pz_bp, pz_hp = bp[0, p0:p1], hp[0, p0:p1]
    sh_bp, sh_hp = bp[1:, s0:s1], hp[1:, s0:s1]

    fc = {}
    for tag, win, band, travel, q0 in (
        ("pz", pz_hp, cfg.p_band, tp, cfg.q0 * cfg.qp_over_qs),
        ("sh", sh_hp, cfg.s_band, ts, cfg.q0),
    ):
        f, pw = F.mt_power(win, fs)
        _put(row, f"f_{tag}_", F.spectral_shape(f, pw, *band))
        _put(row, f"f_{tag}_", F.source_fit(f, pw, *band))
        row[f"f_{tag}_snoke_fc_hz"] = F.snoke_fc(f, pw, *band)
        pc = F.attenuation_correct(f, pw, max(travel, 0.0), q0, cfg.qn)
        src = F.source_fit(f, pc, *band)
        _put(row, f"f_{tag}_qc_", src)
        row[f"f_{tag}_qc_snoke_fc_hz"] = F.snoke_fc(f, pc, *band)
        fc[tag] = (src["brune_fc_hz"], row[f"f_{tag}_qc_snoke_fc_hz"])
    row["f_fc_p_over_s_brune"] = fc["pz"][0] / fc["sh"][0]
    row["f_fc_p_over_s_snoke"] = fc["pz"][1] / fc["sh"][1]

    n_lpdt = min(int(cfg.lpdt_len * fs), i_s - int(0.2 * fs) - ip)
    if n_lpdt >= int(0.5 * fs):
        _put(row, "f_lpdt_", F.lpdt(hp[0, ip:ip + n_lpdt], fs))
    else:
        _put(row, "f_lpdt_", dict.fromkeys(("t_plateau_s", "slope_dec_per_s", "t90_s", "fit_rms"), np.nan))
    _put(row, "f_pz_", F.peak_periods(pz_hp, fs))
    _put(row, "f_sh_", F.peak_periods(sh_hp, fs))

    row["f_pz_tau_c"] = F.tau_c(pz_hp, fs)
    row["f_pz_tau_p_max"] = F.tau_p_max(pz_hp, fs)

    _put(row, "f_pz_", F.waveform_stats(pz_bp))
    _put(row, "f_pz_", F.complexity(pz_bp))
    sh = [F.waveform_stats(c) | F.complexity(c) for c in sh_bp]
    _put(row, "f_sh_", {k: np.nanmean([d[k] for d in sh]) for k in sh[0]})

    q = int(cfg.pol_len * fs)
    _put(row, "f_ppol_", F.polarisation(*bp[:, p0:p0 + q]))
    _put(row, "f_spol_", F.polarisation(*bp[:, s0:s0 + q]))
    row["f_log_s_over_p"] = math.log10(_rms(bp[:, s0:s1]) / _rms(bp[:, p0:p1]))

    w = bp[:, e0:e1]
    envz = F.envelope(w[0], fs)
    envh = np.sqrt(F.envelope(w[1], fs) ** 2 + F.envelope(w[2], fs) ** 2)
    envh /= envh.max()
    t_env0 = t0 + e0 / fs
    ip_e, is_e = ip - e0, i_s - e0
    _put(row, "f_envz_", F.envelope_shape(envz, fs, ip_e, is_e, t_env0))
    _put(row, "f_envh_", F.envelope_shape(envh, fs, ip_e, is_e, t_env0))
    # Durations from P onwards (the env window's pre-P second would only add noise).
    _put(row, "f_durz_", F.significant_duration(bp[0, ip:e1], fs))
    _put(row, "f_durh_", F.significant_duration(bp[1:, ip:e1], fs))
    return row, "ok"


# --------------------------------------------------------------------------
# Event level
# --------------------------------------------------------------------------

COMP_ALIASES = {"Z": "Z", "N": "N", "E": "E", "1": "N", "2": "E"}
BAND_PREFERENCE = ("HH", "BH", "EH", "HN")


def _station_arrays(st, station: str):
    """Pick one instrument for ``station`` and return merged (Z, N, E) arrays."""
    sub = st.select(station=station)
    for band in BAND_PREFERENCE:
        comps = {}
        for tr in sub:
            if tr.stats.channel[:2] == band and tr.stats.channel[-1] in COMP_ALIASES:
                comps.setdefault(COMP_ALIASES[tr.stats.channel[-1]], []).append(tr)
        if set(comps) >= {"Z", "N", "E"}:
            break
    else:
        return None, "missing_components"

    from obspy import Stream

    out = {}
    for c in "ZNE":
        s = Stream(comps[c])
        if len({tr.stats.sampling_rate for tr in s}) != 1:
            return None, "mixed_sampling_rate"
        s.merge(method=1, fill_value=None)
        out[c] = s[0]
    fs = out["Z"].stats.sampling_rate
    if any(out[c].stats.sampling_rate != fs for c in "NE"):
        return None, "mixed_sampling_rate"
    start = max(tr.stats.starttime for tr in out.values())
    end = min(tr.stats.endtime for tr in out.values())
    if end - start < 30:
        return None, "short_record"
    arrs = []
    for c in "ZNE":
        tr = out[c].slice(start, end)
        d = np.ma.filled(np.ma.asarray(tr.data, dtype=np.float64), np.nan)
        arrs.append(d)
    m = min(len(a) for a in arrs)
    return (np.vstack([a[:m] for a in arrs]), fs, start, band), "ok"


def process_event(path: Path, event_id: int, ev: dict, stations: dict,
                  tt: TravelTimes, cfg: Config = Config()) -> tuple[list[dict], dict]:
    """All usable station records of one event file, plus QC reason counts."""
    from obspy import UTCDateTime, read

    qc: dict[str, int] = {}
    try:
        st = read(str(path))
    except Exception:
        return [], {"unreadable_file": 1}
    origin = UTCDateTime(ev["time"].isoformat())
    rows = []
    for sta in sorted({tr.stats.station for tr in st}):
        reason = "ok"
        if sta not in stations:
            reason = "no_station_coords"
        else:
            got, reason = _station_arrays(st, sta)
        if reason == "ok":
            data, fs, start, band = got
            slat, slon, selev = stations[sta]
            epi = epicentral_km(ev["lat"], ev["lon"], slat, slon)
            depth = max(ev["depth_km"], 0.0)
            hypo = math.hypot(epi, depth + selev / 1000.0)
            tp, ts = tt(depth, epi)
            row, reason = record_features(*data, fs=fs, start_s=start - origin,
                                          tp_theo=tp, ts_theo=ts, cfg=cfg)
        qc[reason] = qc.get(reason, 0) + 1
        if reason != "ok":
            continue
        net = st.select(station=sta)[0].stats.network
        row.update(
            event_id=event_id, network=net, station=sta, band=band,
            mag=ev["mag"], mag_type=ev["mag_type"],
            ev_lat=ev["lat"], ev_lon=ev["lon"], time=ev["time"],
            ctl_epi_km=epi, ctl_hypo_km=hypo, ctl_depth_km=ev["depth_km"],
            ctl_baz_deg=back_azimuth(ev["lat"], ev["lon"], slat, slon),
            ctl_sp_s=ts - tp,
        )
        rows.append(row)
    return rows, qc


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
