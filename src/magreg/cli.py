"""``magreg extract | select | audit | train``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

DATA = Path("data")


def cmd_extract(a):
    from joblib import Parallel, delayed

    from .arrivals import TravelTimes
    from .catalog import index_waveforms, load_catalog, load_stations
    from .extract import Config, config_dict, process_event

    cat = load_catalog(a.catalog)
    stations = load_stations(a.stations)
    files = index_waveforms(a.waveforms)
    ids = sorted(set(files) & set(cat.index))
    if a.events:
        wanted = set(pd.read_csv(a.events)["event_id"].astype(int))
        ids = [i for i in ids if i in wanted]
    if a.limit:
        ids = ids[: a.limit]
    print(f"{len(ids)} events with waveforms and catalogue entries")

    cfg = Config(min_snr=a.min_snr)
    tt = TravelTimes(a.tt_cache)  # build the table once, before forking

    def job(i):
        return process_event(files[i], i, cat.loc[i].to_dict(), stations, tt, cfg)

    t0 = time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5, batch_size=8)(delayed(job)(i) for i in ids)
    rows, qc = [], {}
    for r, q in res:
        rows.extend(r)
        for k, v in q.items():
            qc[k] = qc.get(k, 0) + v
    df = pd.DataFrame(rows)
    front = ["event_id", "network", "station", "band", "time", "mag", "mag_type", "ev_lat", "ev_lon"]
    df = df[front + sorted(c for c in df.columns if c not in front)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    meta = {"config": config_dict(cfg), "qc": qc, "events": len(ids),
            "records": len(df), "seconds": round(time.time() - t0, 1)}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["qc"], indent=2))
    nf = sum(c.startswith("f_") for c in df.columns)
    print(f"{len(df)} records, {df['event_id'].nunique() if len(df) else 0} events, "
          f"{nf} features -> {a.out}  ({meta['seconds']} s)")


def cmd_select(a):
    from .catalog import load_catalog
    from .select import balanced_selection, summary

    if a.features:
        df = pd.read_parquet(a.features, columns=["event_id", "time", "mag", "mag_type", "ev_lat", "ev_lon"])
        n = df.groupby("event_id").size()
        ev = (df.drop_duplicates("event_id").set_index("event_id")
              .rename(columns={"ev_lat": "lat", "ev_lon": "lon"}))
        ev["n_records"] = n
        ev = ev[ev["n_records"] >= a.min_records]
    else:
        ev = load_catalog(a.catalog)
    if a.mag_types:
        ev = ev[ev["mag_type"].isin(a.mag_types)]
    ev = ev[ev["mag"] >= a.min_mag]
    chosen = balanced_selection(ev, cap=a.cap, width=a.width, origin=a.min_mag,
                                cell_deg=a.cell_deg, seed=a.seed,
                                max_cell_share=a.max_cell_share)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    chosen.reset_index().to_csv(a.out, index=False)
    with pd.option_context("display.max_rows", None):
        print(summary(ev, chosen, a.width, a.min_mag))
    print(f"{len(chosen)} of {len(ev)} eligible events -> {a.out}")


def cmd_audit(a):
    from .audit import leak_table

    df = pd.read_parquet(a.features)
    if a.events:
        df = df[df["event_id"].isin(pd.read_csv(a.events)["event_id"])]
    t = leak_table(df)
    t.to_csv(a.out)
    with pd.option_context("display.max_rows", None, "display.width", 160,
                           "display.float_format", "{:+.3f}".format):
        print(t)
    print(f"-> {a.out}")


def cmd_train(a):
    from .train import FEATURE_SETS, TrainConfig, run

    df = pd.read_parquet(a.features)
    if a.selection and Path(a.selection).exists():
        keep = pd.read_csv(a.selection)["event_id"]
        df = df[df["event_id"].isin(keep)]
        print(f"selection {a.selection}: {df['event_id'].nunique()} events")
    elif a.selection:
        raise SystemExit(f"{a.selection} not found; run `magreg select` or pass --selection ''")
    if a.mag_types:
        df = df[df["mag_type"].isin(a.mag_types)]
    cfg = TrainConfig(split=a.split, folds=a.folds, repeats=a.repeats, seed=a.seed,
                      weight=a.weight, tail_weight=a.tail_weight,
                      event_medians=not a.no_event_medians)
    print(f"{len(df)} records, {df['event_id'].nunique()} events, {df['station'].nunique()} stations")
    res = run(df, cfg, sets=a.sets or FEATURE_SETS, out=a.out,
              include_leaky=a.include_leaky, leak_threshold=a.leak_threshold)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n== event-level, mean over folds ==")
        print(res["summary"])
        print(f"constant (train mean): {res['constant_mae']:.3f}   "
              f"gap ctl->amplitude closed by shape: {res['gap_closed']:.1%}")
        print("\n== feature families (shape model): event-MAE increase when shuffled ==")
        print(res["families"])
        print("\n== physical quantities (station value + event median shuffled together) ==")
        print(res["quantities"].head(25))
        print("\n== concepts (substitutable quantities shuffled jointly) ==")
        print(res["concepts"].drop(columns="members"))
        print("\n== top 25 features by mean |SHAP| ==")
        print(res["shap"].head(25))
        print("\n== shape model by magnitude ==")
        print(res["by_mag"])
    print(f"-> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="magreg")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="waveforms -> per-record feature table")
    e.add_argument("--catalog", type=Path, default=DATA / "catalogs/catalog_afad_full_2026-08-30.csv")
    e.add_argument("--stations", type=Path, default=DATA / "catalogs/station_coords.csv")
    e.add_argument("--waveforms", type=Path, default=DATA / "waveforms/window_m60_p120")
    e.add_argument("--tt-cache", type=Path, default=DATA / "cache/iasp91_tt.npz")
    e.add_argument("--events", type=Path, help="CSV with an event_id column to restrict to")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--min-snr", type=float, default=3.0)
    e.add_argument("-j", "--jobs", type=int, default=-1)
    e.add_argument("-o", "--out", type=Path, default=DATA / "features/features.parquet")
    e.set_defaults(fn=cmd_extract)

    s = sub.add_parser("select", help="magnitude-balanced event selection")
    s.add_argument("--features", default=str(DATA / "features/features.parquet"),
                   help="restrict to events with QC-passed records (pass '' to use the catalogue)")
    s.add_argument("--catalog", type=Path, default=DATA / "catalogs/catalog_afad_full_2026-08-30.csv")
    s.add_argument("--cap", type=int, default=500, help="max events per magnitude bin")
    s.add_argument("--width", type=float, default=0.2)
    s.add_argument("--min-mag", type=float, default=2.0)
    s.add_argument("--min-records", type=int, default=3)
    s.add_argument("--mag-types", nargs="*", default=None, help="e.g. ML")
    s.add_argument("--cell-deg", type=float, default=1.0)
    s.add_argument("--max-cell-share", type=float, default=0.2,
                   help="max fraction of a magnitude bin from one space-time cell (1 = off)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-o", "--out", type=Path, default=DATA / "features/selection.csv")
    s.set_defaults(fn=cmd_select)

    u = sub.add_parser("audit", help="partial correlation of features with SNR")
    u.add_argument("--features", type=Path, default=DATA / "features/features.parquet")
    u.add_argument("--events", type=Path, help="selection CSV to restrict to")
    u.add_argument("-o", "--out", type=Path, default=DATA / "features/leak_audit.csv")
    u.set_defaults(fn=cmd_audit)

    t = sub.add_parser("train", help="LightGBM + baselines + importance")
    t.add_argument("--features", type=Path, default=DATA / "features/features.parquet")
    t.add_argument("--selection", default=str(DATA / "features/selection.csv"),
                   help="event list from `magreg select` ('' = all events)")
    t.add_argument("--mag-types", nargs="*", default=None)
    t.add_argument("--sets", nargs="*", choices=["shape", "shape_only", "ctl", "amplitude"])
    t.add_argument("--split", choices=["both", "event"], default="both")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--repeats", type=int, default=3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--weight", choices=["event", "none"], default="event")
    t.add_argument("--tail-weight", action="store_true", help="also flatten magnitude histogram")
    t.add_argument("--no-event-medians", action="store_true", help="skip f_evmed_* features")
    t.add_argument("--leak-threshold", type=float, default=0.15,
                   help="exclude f_ features whose within-event |partial rho| with SNR exceeds this")
    t.add_argument("--include-leaky", action="store_true",
                   help="keep features that track SNR (amplitude leaks in; for comparison only)")
    t.add_argument("-o", "--out", type=Path, default=Path("runs/default"))
    t.set_defaults(fn=cmd_train)

    a = p.parse_args(argv)
    if a.cmd == "select":
        a.features = Path(a.features) if a.features else None
    a.fn(a)
