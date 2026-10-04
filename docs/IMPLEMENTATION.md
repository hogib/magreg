# magreg — implementation notes

Goal: estimate earthquake magnitude from waveform features that carry **no
amplitude information**, with LightGBM, and rank which features carry the
magnitude signal. The interesting result is *what* survives once amplitude is
removed, not the raw MAE.

```
uv run magreg extract           # waveforms -> data/features/features.parquet (+ .json QC/config)
uv run magreg select            # balanced event list -> data/features/selection.csv
uv run magreg audit --events data/features/selection.csv   # SNR leak table
uv run pytest
```

## Data

| path | what |
|---|---|
| `data/waveforms/window_m60_p120/event_<id>_raw.mseed` | KO network, mostly HH 100 Hz, origin −60 s … +120 s, raw counts, ~10 stations/event |
| `data/catalogs/catalog_afad_full_2026-08-30.csv` | AFAD catalogue, UTC, **whole-second** origin times |
| `data/catalogs/station_coords.csv` | KO station coordinates (symlink into `sismokaos/data_downloader`) |
| `data/cache/iasp91_tt.npz` | travel-time table, built on first run (~2.5 min) |

12,617 events; M2.0–6.1, median 2.3; 12,257 ML + 360 MW.

## What counts as a magnitude clue

Removing the absolute level is necessary but not sufficient. Excluded from
the model inputs by design:

* **absolute amplitude / energy** in any form;
* **anything relative to pre-event noise** — SNR, STA/LTA, duration above
  noise, coda length to noise. Noise is ~constant, so these are amplitude
  meters (coda duration *is* the Md definition);
* **clipping** — only big events clip; clipped records are dropped;
* **station count per event** — larger events are recorded more widely
  (≈9 vs ≈19 stations for M2–2.5 vs M3–4 in an 800-event sample);
* **catalogue `Type`** — MW rows average M4.05, ML rows M2.42.

Distance is *not* hidden: it is supplied as a control (`ctl_`), because it
both correlates with magnitude through recording selection and changes
waveform shape through attenuation. Without it, shape features would absorb
distance and their importances would be misread as source physics.

## Output table

One row per event × station that passes QC. Column prefixes:

| prefix | role |
|---|---|
| `f_` | amplitude-invariant features — the model inputs |
| `ctl_` | `epi_km`, `hypo_km`, `depth_km`, `baz_deg`, `sp_s` — path controls, also model inputs |
| `ref_` | `log10_peak_counts`, `log10_snr_p/s`, `clipped` — leaky, for baselines and the audit only |
| none | `event_id`, `network`, `station`, `band`, `time`, `mag`, `mag_type`, `ev_lat/lon`, `pick_shift_s` |

`features.json` next to the parquet records the `Config` used and QC reject
counts by reason.

## Pipeline (`extract.py`)

Per event file, per station:

1. **Instrument choice.** First band of `HH, BH, EH, HN` with Z/N/E (1/2 mapped
   to N/E). Segments merged; gaps become NaN and any NaN inside the needed
   span rejects the record (`gap`).
2. **Theoretical arrivals.** First P and S of iasp91 from a cached
   (depth 0–200 km × distance 0–12°) TauP table, bilinear interpolation
   (`arrivals.py`).
3. **Preprocess** the span `[P−8.5 s, S+11 s]`: demean, linear detrend, 5%
   cosine taper, then two causal 4-pole Butterworth versions — band-pass
   1–40 Hz (time-domain features, picking) and high-pass 0.5 Hz (spectra,
   τc). Causal filters keep energy from leaking ahead of the onset. No
   instrument correction: there is no inventory, and broadband HH response
   is flat over the bands used; any per-station gain cancels in
   scale-invariant features anyway.
4. **P pick.** Maeda AIC on band-passed Z in `[P_theo−1.5, min(P_theo+3, S_theo−0.3)]`.
   The window is asymmetric because AFAD origin times are truncated to whole
   seconds: on a 200-event trial the pick shift was a near-constant
   +0.75 s median at every distance (not a velocity-model trend), and the
   picker is unbiased on synthetics. `pick_shift_s` is kept for diagnostics.
   S is placed at `P_pick + (S_theo − P_theo)`.
5. **Windows** (relative to the picks):

   | window | span | components |
   |---|---|---|
   | noise (QC only) | P−6 … P−1 s | all |
   | P | P−0.05 … P+1.95 s (200 samples) | Z |
   | S | S−0.2 … S+4.8 s (500 samples) | N, E |
   | polarisation | first 1 s of P and of S | Z, N, E |
   | envelope | P−1 … S+10 s | Z, and H = √(envN² + envE²) |

   Lengths are fixed so spectral resolution and statistics are comparable
   across records. Envelope windows end at S+10 rather than at a noise
   threshold for the same leak reason as above.
6. **QC** (records dropped, values kept as `ref_`): clipped (|counts| ≥ 8e6
   or ≥5 samples at the peak value), SNR_P < 3 on Z, SNR_S < 3 on N/E. The
   gate is a *selection*, not a feature, but it is magnitude-dependent:
   small events lose distant stations. See "Known residual leaks".

## Features (`features.py`)

Every function first normalises its window by its own peak (`unit`).
Prefixes: `f_pz_` P/Z, `f_sh_` S/horizontals (N and E averaged), `f_ppol_`,
`f_spol_`, `f_envz_`, `f_envh_`.

**Spectral shape** — multitaper (NW 2.5, 4 tapers) velocity power; P band
1–40 Hz, S band 0.7–40 Hz. Log-frequency centroid and bandwidth, flatness,
normalised entropy, log–log slope, peak frequency, roll-off at 50/85/95%,
energy fractions in octave bands 1–2 … 32–45 Hz.

**Source fits** on the displacement amplitude spectrum, resampled to 40
log-spaced points so high frequencies do not dominate:
* Brune + attenuation: `log A = log Ω0 − log(1+(f/fc)²) − πκf` → `brune_fc_hz`,
  `brune_kappa`, `brune_rms`;
* Boatwright, free falloff: `log A = log Ω0 − log(1+(f/fc)^n)` → `boat_fc_hz`,
  `boat_n`, `boat_rms`.

Ω0 is fitted and discarded — it is the level. fc grid 0.3–45 Hz (80 log
steps), n 1–4; `fc_at_edge` flags fits stuck at a grid boundary. Expected
physics: fc ∝ M0^(−1/3), but for M2–3 the corner (~10–30 Hz) overlaps κ and
the anti-alias roll-off, so fc and κ trade off.

**Early-warning periods** (P/Z, high-passed): Kanamori τc over the P window;
Allen–Kanamori recursive τp (α = 0.99), maximum after the first 0.5 s.

**Waveform statistics**: kurtosis, |skewness| (sign is polarity), crest
factor, zero-crossing rate, Hjorth mobility and complexity (Hjorth *activity*
is variance, i.e. amplitude, and is left out).

**Complexity**: permutation entropy (order 5), sample entropy (m 2, r 0.2·std),
Higuchi fractal dimension (kmax 10), SVD entropy (order 10). The Lyapunov /
correlation-dimension code in `sismokaos/chaos` is not ported: it is slow
and ill-conditioned on 2–5 s windows.

**Polarisation** (first second of P and S, three components scaled jointly):
rectilinearity, planarity, incidence angle, log H/V. Plus
`f_log_s_over_p` (3C RMS of S window over P window).

**Envelope shape** (Hilbert, 0.2 s smoothing, peak = 1): time to peak after
P (s and as a fraction of S−P), 10→90% rise time, time above 50% and 20% of
peak, P-window maximum relative to the peak, exponential decay rate and
power-law exponent from the peak to window end, energy centroid time, mean
fill.

## The invariance contract

`tests/test_invariance.py` runs the full record pipeline on a synthetic
3-component record and on copies scaled by 10, 0.01 and −3, and requires all
`f_` columns to agree to 1e-6. Any new feature must pass it. (The −3 case is
what moved skewness to |skewness|.) The same test checks that `ref_` columns
*do* move with scale, AIC onset recovery, Brune fc recovery on an analytic
spectrum, and low-SNR rejection.

## Balanced event selection (`select.py`)

1. **Eligibility**: events with ≥ `--min-records` (3) QC-passed records in the
   feature table, optionally `--mag-types ML`.
2. **Bins** of 0.2 magnitude units from M2.0, each holding exactly two
   0.1-resolution catalogue values (0.25 bins would alternate 3/2/3 values).
3. **Cap** each bin at `--cap` events (default 500). Smaller bins keep
   everything; nothing is duplicated — oversampling ~170 M>4 events only
   invites memorisation. Use sample weights at training time for the tail.
4. **Spread** inside a capped bin: group events into cells of 1°×1°×year and
   draw round-robin across cells in random order. Small events come in swarms;
   a uniform draw lets a few sequences (one region, one path, one station
   set) fill the low bins, and the model could then separate magnitudes by
   region. The test catalogue with a 50% swarm drops to <15% swarm share.

The printed summary shows per bin: available, selected, number of cells and
the largest cell's share.

Record-level note: big events have more stations, so even an event-balanced
set is record-imbalanced. Weight records by `1 / n_records(event)` or
aggregate predictions per event.

## Leak audit (`audit.py`)

For each `f_` feature: Spearman with magnitude, partial Spearman with
magnitude given log distance, and **partial Spearman with log SNR (P and S)
given magnitude and log distance**. At fixed magnitude and distance, SNR
varies only through station noise, site gain and radiation; a feature that
tracks it is reading the noise floor. `leak_score` is the larger of the two
SNR partials; the table is sorted by it.

## Known residual leaks

* A fixed-length window on a weak event contains more noise: spectra flatten,
  envelopes plateau, so shape features can encode SNR. The SNR ≥ 3 gate
  limits but does not remove this — read the audit, and consider dropping
  features with high `leak_score` or evaluating on SNR-matched subsets.
* The SNR gate removes distant records of small events, so the distance
  distribution differs across magnitudes. `ctl_` columns let the model
  condition on it; baselines should get the same controls.
* Magnitude labels mix ML (below ~M4) and MW (above); run an ML-only variant.

## Performance

Python/numpy, joblib over event files. ~14 s per 200 events on 12 cores, so
the full set takes ~15 min; travel-time table build is a one-off ~2.5 min.
No compiled code was needed.
