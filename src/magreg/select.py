"""Magnitude-balanced event selection.

The catalogue is ~70% M2–3. Method:

1. **Bin** magnitudes in fixed-width bins (default 0.2, aligned so each bin
   holds exactly two 0.1-resolution catalogue values).
2. **Cap** every bin at ``cap`` events; bins with fewer keep all of theirs.
   The histogram is flat up to the point where the catalogue runs out, then
   follows the natural tail. Nothing is duplicated: oversampling the M>4 tail
   would just let the model memorise ~170 events. Any remaining imbalance is
   better handled with sample weights at training time.
3. **Spread** within a capped bin: events are grouped into space–time cells
   (``cell_deg`` lat/lon squares × calendar year) and drawn round-robin across
   cells in random order. Small events cluster in swarms and aftershock
   sequences; a uniform draw would let a few sequences (one source region, one
   path, one set of stations) dominate the low-magnitude bins, and the model
   could tell magnitudes apart by *where* the events are rather than by their
   waveforms.

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


def balanced_selection(
    events: pd.DataFrame,
    cap: int,
    width: float = 0.2,
    origin: float = 2.0,
    cell_deg: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Return the selected subset of ``events`` (needs mag, lat, lon, time).

    Adds ``mag_bin`` (bin lower edge) and ``cell`` columns.
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
        if len(grp) <= cap:
            keep.append(grp.index.to_numpy())
            continue
        cells = [rng.permutation(g.index.to_numpy()) for _, g in grp.groupby("cell")]
        order = rng.permutation(len(cells))
        cells = [cells[i] for i in order]
        picked: list = []
        depth = 0
        while len(picked) < cap:
            for c in cells:
                if depth < len(c):
                    picked.append(c[depth])
                    if len(picked) == cap:
                        break
            depth += 1
        keep.append(np.asarray(picked))
    return ev.loc[np.concatenate(keep)].sort_values("time")


def summary(all_events: pd.DataFrame, chosen: pd.DataFrame, width: float, origin: float) -> pd.DataFrame:
    a = all_events.groupby(np.round(origin + mag_bin(all_events["mag"], width, origin) * width, 6)).size()
    c = chosen.groupby("mag_bin").size()
    cells = chosen.groupby("mag_bin")["cell"].nunique()
    top = chosen.groupby("mag_bin")["cell"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    out = pd.DataFrame({"available": a, "selected": c, "cells": cells, "largest_cell_share": top})
    out.index.name = "mag_bin"
    return out.fillna(0).astype({"available": int, "selected": int, "cells": int})
