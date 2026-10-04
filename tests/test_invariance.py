"""The contract: f_ features do not change when the record is rescaled."""

import numpy as np
import pytest

from magreg import features as F
from magreg.arrivals import aic_pick
from magreg.extract import record_features

FS = 100.0
START = -60.0  # record starts 60 s before origin, like window_m60_p120
TP, TS = 8.0, 14.0


def synthetic(seed=0, amp=2000.0):
    """Noise plus a P and an S wavelet train with decaying coda, in counts."""
    rng = np.random.default_rng(seed)
    n = int(180 * FS)
    t = START + np.arange(n) / FS
    out = []
    for c, (ap, as_) in enumerate([(1.0, 0.6), (0.4, 1.0), (0.3, 0.9)]):
        x = rng.normal(0, 20.0, n)
        for t0, a, f0 in [(TP, ap, 7.0), (TS, as_ * 2.5, 4.0)]:
            m = t >= t0
            tt = t[m] - t0
            x[m] += amp * a * np.exp(-tt / 3.0) * np.sin(2 * np.pi * f0 * tt + c) * (1 - np.exp(-tt / 0.05))
            x[m] += amp * a * 0.3 * np.exp(-tt / 6.0) * rng.normal(0, 1, m.sum())
        out.append(x + 500.0)  # DC offset like raw counts
    return out


def features_of(scale, seed=0):
    z, n, e = (scale * c for c in synthetic(seed))
    row, why = record_features(z, n, e, FS, START, TP, TS)
    assert why == "ok", why
    return row


@pytest.mark.parametrize("scale", [10.0, 0.01, -3.0])
def test_features_are_scale_invariant(scale):
    a, b = features_of(1.0), features_of(scale)
    fcols = [k for k in a if k.startswith("f_")]
    assert len(fcols) > 80
    for k in fcols:
        assert np.isclose(a[k], b[k], rtol=1e-6, atol=1e-9, equal_nan=True), (k, a[k], b[k])


def test_reference_columns_do_see_amplitude():
    a, b = features_of(1.0), features_of(10.0)
    assert np.isclose(b["ref_log10_peak_counts"] - a["ref_log10_peak_counts"], 1.0, atol=1e-6)


def test_aic_finds_onset_and_ignores_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    x[250:] += rng.normal(0, 8, 150)
    i = aic_pick(x)
    assert abs(i - 250) <= 3
    assert aic_pick(1e-4 * x) == i


def test_brune_recovers_corner_frequency():
    f = np.fft.rfftfreq(2000, 1 / FS)[1:]
    for fc in (2.0, 8.0):
        disp = 1.0 / (1 + (f / fc) ** 2)
        p = (disp * 2 * np.pi * f) ** 2  # velocity power
        out = F.source_fit(f, p, 0.5, 40.0)
        assert abs(np.log(out["brune_fc_hz"] / fc)) < 0.1
        assert abs(out["brune_kappa"]) < 1e-3
        assert abs(out["boat_n"] - 2.0) < 0.15


def test_low_snr_record_is_rejected():
    z, n, e = synthetic(amp=5.0)
    row, why = record_features(z, n, e, FS, START, TP, TS)
    assert row is None and why.startswith("low_snr")
