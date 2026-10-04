"""Magnitude-balanced event selection.

The catalogue is ~70% M2–3. Method:

1. **Bin** magnitudes in fixed-width bins (default 0.2, aligned so each bin
   holds exactly two 0.1-resolution catalogue values).
2. **Cap** every bin at ``cap`` events; bins with fewer keep all of theirs.
   The histogram is flat up to the point where the catalogue runs out, then
   follows the natural tail. Nothing is duplicated: oversampling the M>4 tail
   would just let the model memorise ~170 events. Any remaining imbalance is
   better handled with sample weights at training time.
3. **Spread** within a bin: events are grouped into space–time cells
   (``cell_deg`` lat/lon squares × calendar year) and drawn round-robin across
   cells in random order. Small events cluster in swarms and aftershock
   sequences; a uniform draw would let a few sequences (one source region, one
   path, one set of stations) dominate the low-magnitude bins, and the model
   could tell magnitudes apart by *where* the events are rather than by their
   waveforms.
4. **Limit cells** in every bin, including bins under the cap:
   ``max_cell_share`` (CLI default 0.2) bounds the share of a bin that one
   cell may supply. Needed because big events are rarer and cluster in
   sequences: on this catalogue the 2025 Sındırgı sequence (cell
   ``39_28_2025``) gave 24–52% of every bin from M2.8 up and ~1% below, a
   regional signature the model could learn instead of source physics.

Eligibility is applied first: an event only counts if it has at least
``min_records`` records that survived QC. The cap therefore balances what the
model will actually see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mag_bin(m: pd.Series, width: float, origin: float) -> pd.Series:
    # 1e-6 guards values like 2.1999999 sitting on a bin edge.
    return np.floor((m - origin) / width + 1e-6).astype(int)


def _round_robin(grp: pd.DataFrame, rng) -> list:
    """Bin events ordered by cycling over space–time cells in random order."""
    cells = [rng.permutation(g.index.to_numpy()) for _, g in grp.groupby("cell")]
    cells = [cells[i] for i in rng.permutation(len(cells))]
    out = []
    for depth in range(max(len(c) for c in cells)):
        out.extend(c[depth] for c in cells if depth < len(c))
    return out


def _take(order: list, cells: pd.Series, cap: int, per_cell: int) -> list:
    taken, count = [], {}
    for i in order:
        c = cells[i]
        if count.get(c, 0) < per_cell:
            taken.append(i)
            count[c] = count.get(c, 0) + 1
            if len(taken) == cap:
                break
    return taken


def balanced_selection(
    events: pd.DataFrame,
    cap: int,
    width: float = 0.2,
    origin: float = 2.0,
    cell_deg: float = 1.0,
    seed: int = 0,
    max_cell_share: float = 1.0,
    min_cell_cap: int = 3,
) -> pd.DataFrame:
    """Return the selected subset of ``events`` (needs mag, lat, lon, time).

    ``max_cell_share`` limits any one space–time cell to that fraction of
    its bin's final count (never below ``min_cell_cap`` events), including
    bins under the cap — this is what stops one productive sequence from
    supplying most of the large events. Adds ``mag_bin`` and ``cell``.
    """
    rng = np.random.default_rng(seed)
    ev = events.copy()
    b = mag_bin(ev["mag"], width, origin)
    ev["mag_bin"] = np.round(origin + b * width, 6)
    ev["cell"] = (
        np.floor(ev["lat"] / cell_deg).astype(int).astype(str)
        + "_" + np.floor(ev["lon"] / cell_deg).astype(int).astype(str)
        + "_" + pd.to_datetime(ev["time"]).dt.year.astype(str)
    )
    keep = []
    for _, grp in ev.groupby("mag_bin", sort=True):
        order = _round_robin(grp, rng)
        per_cell = len(order)
        taken = _take(order, ev["cell"], cap, per_cell)
        # Shrink the per-cell limit until it satisfies the share on the final
        # count (dropping events lowers the count, which lowers the limit).
        while max_cell_share < 1.0:
            lim = max(min_cell_cap, int(np.ceil(max_cell_share * len(taken))))
            if lim >= per_cell:
                break
            per_cell = lim
            taken = _take(order, ev["cell"], cap, per_cell)
        keep.append(np.asarray(taken))
    return ev.loc[np.concatenate(keep)].sort_values("time")


def summary(all_events: pd.DataFrame, chosen: pd.DataFrame, width: float, origin: float) -> pd.DataFrame:
    a = all_events.groupby(np.round(origin + mag_bin(all_events["mag"], width, origin) * width, 6)).size()
    c = chosen.groupby("mag_bin").size()
    cells = chosen.groupby("mag_bin")["cell"].nunique()
    top = chosen.groupby("mag_bin")["cell"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    out = pd.DataFrame({"available": a, "selected": c, "cells": cells, "largest_cell_share": top})
    out.index.name = "mag_bin"
    return out.fillna(0).astype({"available": int, "selected": int, "cells": int})
