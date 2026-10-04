# Results log

## 2026-10-05 — first runs

Data: 78,732 QC-passed records from 11,758 events. Selection: cap 500 per
0.2 bin, max cell share 0.2, ≥3 records per event, giving **3,697 events,
31,476 records, 151 stations**. Split `both` (event- and station-disjoint),
5 folds × 3 repeats; weights `event`. Event-level MAE, mean ± sd over 15
folds.

| feature set | `runs/all` (147 features; now needs `--include-leaky`) | `runs/clean015` (61, leak > 0.15 dropped; now the default) |
|---|---|---|
| constant (train mean) | 0.466 | 0.466 |
| `ctl` (distance, depth) | 0.392 ± 0.016 | same |
| **`shape`** | **0.320 ± 0.024** | **0.343 ± 0.017** |
| `shape_only` | 0.327 ± 0.027 | 0.347 ± 0.018 |
| `amplitude` (peak counts + ctl) | 0.238 ± 0.022 | same |
| gap ctl → amplitude closed | 47% | 32% |

### Reading

* **Shape carries real magnitude information.** It beats distance-only by
  0.05–0.07 MAE with a station-disjoint split. Even after removing every
  feature that tracks SNR within events, it closes about a third of the gap
  to a fitted amplitude relation.
* **About a third of the "all" gain is leak.** Dropping the 79 noise-tracking
  features costs 0.023 MAE (47% → 32% gap closed). The `all` number should
  not be quoted as amplitude-free.
* **The clean model runs on physics.** Top features by |SHAP| in
  `clean015`: event-median attenuation-corrected Snoke corner frequency, P
  (0.18) and S (0.12); epicentral distance (0.12); depth; S Boatwright fit
  residual; event-median LPDT plateau time; P/S corner ratio; LPDT slope.
  Family permutation: event medians 0.097, distance 0.044, then S source
  (attenuation-corrected) 0.012.
* **In `all`, `f_evmed_pz_tau_p_max` is first** (|SHAP| 0.19). τp max has
  leak score 0.18, just over the 0.15 threshold, so part of its value may be
  noise. `f_evmed_sh_qc_brune_kappa` (second, 0.09) is clearly leaky (0.45).
* **Network medians matter most.** Per-station estimates are noisy
  (radiation pattern, site); the median across stations is what works. This
  matches standard practice for source parameters.
* **Distance helps through selection**: `ctl` alone reaches 0.392 against
  0.466 for the constant.

### Bias by magnitude (event bias / MAE)

| M bin | shape (all) | shape (clean) | amplitude | ctl |
|---|---|---|---|---|
| 2.0 | +0.23 / 0.27 | +0.28 / 0.31 | +0.20 / 0.23 | +0.38 / 0.38 |
| 2.5 | −0.06 / 0.27 | −0.08 / 0.26 | −0.03 / 0.20 | −0.09 / 0.16 |
| 3.0 | −0.12 / 0.32 | −0.15 / 0.34 | −0.07 / 0.23 | −0.29 / 0.39 |
| 3.5 | −0.37 / 0.45 | −0.45 / 0.50 | −0.25 / 0.32 | −0.73 / 0.73 |
| 4.0 | −0.56 / 0.60 | −0.67 / 0.71 | −0.39 / 0.42 | −1.16 / 1.16 |
| 4.5 | −0.63 / 0.63 | −0.83 / 0.84 | −0.35 / 0.39 | −1.62 / 1.62 |
| 5.0+ | −0.9 … −1.2 | −1.2 … −1.6 | −0.5 … −0.9 | −2 … −3 |

Every model regresses toward the mean, including the amplitude baseline, so
part of this is the thin tail (~170 events ≥ M4). Shape is worse at the top.
Two likely causes:

1. **The windows are too short for large events.** A 2 s P window has
   0.5 Hz resolution, and the S window 5 s. An M4.5 corner is about 1–2 Hz
   and its source duration about 1 s. Corner estimates for large events are
   pinned near the band floor, and LPDT (≤ 3 s) can't see a plateau.
2. **ML/MW mixing**: above about M4 the labels are mostly MW.

The +0.2 to +0.3 bias at M2.0 is a floor effect: the catalogue stops at
M2.0, so there is nothing below to balance against.

## 2026-10-05 — tail weighting and ML-only

Both runs use the default leak filter, so they compare with `runs/clean015`.

### `runs/tailw` (`--tail-weight`)

| | default (`clean015`) | tail-weighted |
|---|---|---|
| event MAE, all events | 0.343 ± 0.017 | 0.367 ± 0.019 |
| event MAE, MW events (mostly M ≥ 3.8) | 0.687 ± 0.061 | 0.609 ± 0.054 |
| bias M2.0 | +0.28 | +0.33 |
| bias M4.0 / 4.5 / 5.0 | −0.67 / −0.83 / −1.16 | −0.57 / −0.70 / −1.04 |

Up-weighting the tail by up to 10× removes only about 0.1 of the large-event
bias, and costs 0.024 MAE overall plus extra bias at M2.0. **The
underestimation is not mainly a sample-imbalance problem.** The features
don't separate M4 from M5, which fits the short-window explanation below.
Not adopted.

### `runs/mlonly` (`--mag-types ML`)

Removes 246 MW events (median M4.0, range 3.0–6.1), so the ML-only training
set has almost nothing above M4. Compared on the **same ML events**:

| | default model | ML-only model |
|---|---|---|
| event MAE | 0.314 ± 0.015 | 0.302 ± 0.016 |
| bias M2.0 / 2.5 / 3.0 | +0.28 / −0.08 / −0.15 | +0.25 / −0.13 / −0.25 |
| bias M3.5 / 4.0 | −0.42 / −0.69 | −0.57 / −1.02 |

The overall gain (0.012) is within one fold sd, and it comes from the bulk
around M2–2.5. Above M3 the ML-only model compresses *more*, because MW
training events are what teach it the top of the range. In the default run,
MW events in M3.5–4.5 are underestimated more than ML events in the same
range (−0.61 vs −0.43). That may be the scale difference, or just that MW
events sit higher within the band, so it's inconclusive. **Mixing ML and MW
is not what limits the model.** Keep both.

## 2026-10-05 — per-quantity permutation importance (`runs/default`)

Default configuration (leaky features excluded; same scores as `clean015`:
0.343 ± 0.017). Each physical quantity — the per-station column and its event
median together — is shuffled with one shared row permutation, 3 shuffles ×
15 folds; the score is the rise in event-level MAE.

| rank | quantity | ΔMAE | sd | folds > 0 | what it is |
|---|---|---|---|---|---|
| 1 | `ctl_epi_km` | 0.041 | 0.009 | 15/15 | distance (control) |
| 2 | **`f_pz_qc_snoke_fc_hz`** | **0.019** | 0.008 | 15/15 | P corner frequency, attenuation-corrected |
| 3 | `f_sh_qc_boat_rms` | 0.016 | 0.010 | 15/15 | misfit of the ω-model to the corrected S spectrum |
| 4 | `f_sh_qc_snoke_fc_hz` | 0.013 | 0.009 | 14/15 | S corner frequency, attenuation-corrected |
| 5 | `f_lpdt_t_plateau_s` | 0.008 | 0.006 | 15/15 | P displacement plateau time (rupture duration) |
| 6 | `f_pz_frac_1_2` | 0.008 | 0.002 | 15/15 | P energy fraction 1–2 Hz |
| 7 | `f_sh_boat_rms` | 0.007 | 0.004 | 15/15 | same misfit, uncorrected |
| 8 | `ctl_depth_km` | 0.006 | 0.006 | 12/15 | depth (control) |

Only 8 of 57 quantities hurt the model in every fold when shuffled.

### Reading

* **The most important waveform quantity is the P-wave corner frequency**,
  corrected for attenuation and taken as the median across stations. It agrees
  with the SHAP ranking (first, 16.5% of |SHAP|).
* **Distance comes first in permutation, but SHAP puts it third.** The
  difference is substitution: the P and S corner frequencies are correlated
  (both ρ ≈ −0.27 with magnitude), so shuffling one leaves the other to cover
  for it, and each individual ΔMAE understates the pair. Distance has no
  substitute. Its role is the selection effect (big events recorded farther
  away) plus conditioning the attenuation-sensitive features.
* **`f_sh_qc_boat_rms` is a quality signal, not a size signal.** Its own
  correlation with magnitude is ≈ 0 (ρ 0.01 given distance; leak 0.14, under
  the threshold). It tells the trees how far to trust the corner-frequency
  estimate on a given record. Interpret it as an interaction, not a magnitude
  carrier.
* **LPDT plateau time ranks next**, the only time-domain duration measure
  that holds up. Durations from the Husid curve, envelopes and complexity
  measures each contribute ≲ 0.005.

### Concepts shuffled jointly (`importance_concepts.csv`)

Substitutable quantities shuffled together, so neither can cover for the other:

| concept | members | ΔMAE | sd | folds > 0 |
|---|---|---|---|---|
| **corner frequency, P+S (Snoke, corrected)** | `pz_qc_snoke_fc`, `sh_qc_snoke_fc` | **0.086** | 0.012 | 15/15 |
| all corner-frequency estimates | the two above, uncorrected Snoke P/S, P/S ratios | 0.084 | 0.012 | 15/15 |
| all distance controls | `epi_km`, `hypo_km`, `sp_s` | 0.073 | 0.015 | 15/15 |
| rupture duration (LPDT) | plateau time, slope, t90 | 0.008 | 0.004 | 15/15 |

(Brune and Boatwright corner estimates are not in `corner_freq_all`
because the leak filter drops them.)

**The corner frequency is the most important quantity in the model, ahead
of distance.** Shuffled together, the P and S corners cost 0.086 MAE, against
0.019 + 0.013 when shuffled one at a time. That is the substitution the
single-quantity test hides. Adding every other corner estimate barely
changes it (0.084), so the attenuation-corrected Snoke pair carries the
whole corner-frequency signal. Shuffling feeds the model wrong values rather
than removing the feature, so 0.086 is larger than the 0.049 gap between
`shape` and `ctl`. It measures reliance, not the drop you'd see from
retraining without the feature.

## Next

* Longer, magnitude-agnostic windows for source fits (fixed 4 s P / 10 s S,
  or several lengths as separate features), then check the tail bias again.
* Re-check τp max at a threshold of 0.2 to settle whether its importance is
  physics or noise.
