"""AFAD catalogue, KO station coordinates and the waveform file index."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_FILE = re.compile(r"event_(\d+)_raw\.mseed$")


def load_catalog(path: Path) -> pd.DataFrame:
    """AFAD CSV -> one row per event, indexed by ``event_id``.

    Origin times are UTC (checked against the -60 s start of the waveform
    windows).
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    out = pd.DataFrame(
        {
            "event_id": df["EventID"].astype(np.int64),
            "time": pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M:%S", utc=True),
            "lat": df["Latitude"].astype(float),
            "lon": df["Longitude"].astype(float),
            "depth_km": df["Depth"].astype(float),
            "mag": df["Magnitude"].astype(float),
            "mag_type": df["Type"].astype(str),
        }
    )
    return out.drop_duplicates("event_id").set_index("event_id")


def load_stations(path: Path) -> dict[str, tuple[float, float, float]]:
    """``station -> (lat, lon, elevation_m)``."""
    df = pd.read_csv(path)
    return {
        r.station: (float(r.latitude), float(r.longitude), float(r.elevation))
        for r in df.itertuples()
    }


def index_waveforms(directory: Path) -> dict[int, Path]:
    """``event_id -> mseed path`` for every ``event_<id>_raw.mseed``."""
    out = {}
    for p in Path(directory).iterdir():
        m = EVENT_FILE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def epicentral_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km (haversine)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * 6371.0 * np.arcsin(np.sqrt(a)))


def back_azimuth(ev_lat, ev_lon, st_lat, st_lon) -> float:
    """Azimuth from station to event, degrees clockwise from north."""
    p1, p2 = np.radians(st_lat), np.radians(ev_lat)
    dl = np.radians(ev_lon - st_lon)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return float(np.degrees(np.arctan2(y, x)) % 360.0)
