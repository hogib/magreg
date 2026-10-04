import numpy as np

from magreg import features as F

FS = 100.0


def brune_velocity_power(f, fc):
    return (2 * np.pi * f / (1 + (f / fc) ** 2)) ** 2


def test_snoke_fc_recovers_brune_corner():
    f = np.fft.rfftfreq(20000, 1 / FS)[1:]
    for fc in (2.0, 5.0):
        est = F.snoke_fc(f, brune_velocity_power(f, fc), 0.05, 50.0)
        assert abs(est / fc - 1) < 0.15


def test_attenuation_correction_undoes_attenuation():
    f = np.fft.rfftfreq(1000, 1 / FS)[1:]
    p = brune_velocity_power(f, 6.0)
    q = 60 * f**0.85
    att = p * np.exp(-2 * np.pi * f * 20.0 / q)
    back = F.attenuation_correct(f, att, 20.0, 60, 0.85)
    assert np.allclose(back, p)
    # attenuation drags the fitted corner down; correction restores it
    raw = F.source_fit(f, att, 1, 40)["brune_fc_hz"]
    fixed = F.source_fit(f, back, 1, 40)["brune_fc_hz"]
    assert abs(np.log(fixed / 6.0)) < abs(np.log(raw / 6.0)) + 1e-9
    assert abs(np.log(fixed / 6.0)) < 0.1


def test_lpdt_plateau_tracks_pulse_duration():
    out = []
    for dur in (0.3, 1.2):
        t = np.arange(int(3 * FS)) / FS
        u = np.where(t < dur, t / dur, 1.0)  # displacement ramps then holds
        v = np.gradient(u) * FS
        out.append(F.lpdt(v + 1e-4 * np.random.default_rng(0).normal(size=len(v)), FS)["t_plateau_s"])
    assert out[0] < out[1]


def test_significant_duration_of_boxcar():
    x = np.zeros(1000)
    x[200:600] = 1.0
    d = F.significant_duration(x, FS)
    assert abs(d["d5_95_s"] - 3.6) < 0.05
    assert abs(d["t5_s"] - 2.2) < 0.05


def test_peak_periods_of_sinusoid():
    t = np.arange(int(4 * FS)) / FS
    v = np.sin(2 * np.pi * 2.0 * t)
    p = F.peak_periods(v, FS)
    assert abs(p["T_va_s"] - 0.5) < 0.02  # 2π·V/A = 1/f for a sinusoid
