import numpy as np
import pandas as pd

from magreg.select import balanced_selection, mag_bin


def catalog(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    mags = np.round(2.0 + rng.exponential(0.45, n), 1)
    swarm = rng.random(n) < 0.5  # half the events in one 1-degree cell
    return pd.DataFrame({
        "mag": mags,
        "lat": np.where(swarm, 38.5, rng.uniform(36, 42, n)),
        "lon": np.where(swarm, 27.5, rng.uniform(26, 44, n)),
        "time": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.uniform(0, 365, n), "D"),
    })


def test_bins_hold_two_catalogue_values():
    m = pd.Series([2.0, 2.1, 2.2, 2.3, 2.4])
    assert list(mag_bin(m, 0.2, 2.0)) == [0, 0, 1, 1, 2]


def test_cap_flattens_and_keeps_tail():
    ev = catalog()
    sel = balanced_selection(ev, cap=200)
    counts = sel.groupby("mag_bin").size()
    full = ev.groupby(np.round(2.0 + mag_bin(ev["mag"], 0.2, 2.0) * 0.2, 6)).size()
    assert counts.max() == 200
    for b, n in full.items():
        assert counts[b] == min(n, 200)
    assert not sel.index.duplicated().any()


def test_round_robin_limits_swarm_share():
    ev = catalog()
    sel = balanced_selection(ev, cap=100, seed=1)
    low = sel[sel["mag_bin"] == 2.0]
    share = (low["lat"] == 38.5).mean()
    assert share < 0.15  # a uniform draw would give ~0.5


def test_deterministic():
    ev = catalog()
    a = balanced_selection(ev, cap=150, seed=3).index
    b = balanced_selection(ev, cap=150, seed=3).index
    assert a.equals(b)
