# Feature reference

Every model input (`f_` column), how it is computed, what it is supposed to
measure, and why it carries no amplitude. Controls (`ctl_`), leaky references
(`ref_`) and the per-event medians added at training time are at the end.
Implementation: `src/magreg/features.py` (functions), `src/magreg/extract.py`
(windows and wiring), `src/magreg/train.py` (`f_evmed_*`).

131 record-level features + 11 event medians.

## Notation and conventions

| symbol | meaning |
|---|---|
| $f_s$ | sampling rate (100 Hz) |
| $t_P, t_S$ | picked P time; S time $= t_P + (t_S^{th} - t_P^{th})$ (iasp91 S−P), both seconds after origin |
| $x[n]$ | one window of one component, band-passed 1–40 Hz (causal 4-pole Butterworth) unless noted |
| $v[n]$ | same window high-passed at 0.5 Hz — a velocity trace (HH sensors record velocity) |
| $\hat x$ | `unit(x)`: $\hat x = (x - \bar x) / \max\lvert x - \bar x\rvert$ |
| $u[n]$ | displacement, $u = \frac{1}{f_s}\sum v$, minus the straight line joining its end points (integration drift) |
| $P(f)$ | multitaper velocity power spectrum (below) |
| $D(f)$ | displacement amplitude spectrum, $D = \sqrt{P}/(2\pi f)$ |
| $\varepsilon$ | $10^{-12}$, guards divisions and logs |

**Windows** (see `IMPLEMENTATION.md` for the reasoning):

| prefix | window | components |
|---|---|---|
| `f_pz_` | P: $[t_P - 0.05,\ t_P + 1.95)$ s, 200 samples | Z |
| `f_sh_` | S: $[t_S - 0.2,\ t_S + 4.8)$ s, 500 samples | N and E (averaged as stated per feature) |
| `f_ppol_` / `f_spol_` | first 1.0 s of the P / S window | Z, N, E |
| `f_envz_` / `f_envh_` | envelope: $[t_P - 1,\ t_S + 10)$ s | Z / horizontal |
| `f_durz_` / `f_durh_` | $[t_P,\ t_S + 10)$ s | Z / N+E |
| `f_lpdt_` | $[t_P,\ t_P + \min(3, t_S - t_P - 0.2))$ s | Z, high-passed |

**Why everything is level-free.** Multiplying a record by $c \ne 0$ changes $x$,
$v$, $u$, $P$ and $D$ by $c$ or $c^2$. Every feature below is a ratio of
quantities of the same degree in $c$, a time or frequency read off a
normalised curve, an ordering, or a fitted shape whose level parameter is
thrown away. `tests/test_invariance.py` checks all 131 columns at
$c \in \{10, 0.01, -3\}$.

**Leak caveat that applies to all of them.** None uses the pre-event noise.
But a weak record's window is partly noise, so its *shape* (flat spectrum,
envelope that never decays) can still reflect SNR. `magreg audit` measures
this per feature.

---

## 1. Spectral shape — `f_pz_*`, `f_sh_*`

**Spectrum.** Thomson multitaper with 4 DPSS tapers $w_k$, time-bandwidth
$NW = 2.5$, on the high-passed window normalised by its peak:

$$P(f) = \frac{1}{K}\sum_{k=1}^{K} \Big\lvert \mathrm{FFT}\big(w_k\,\hat v\big)(f) \Big\rvert^2$$

For `f_sh_` the N and E spectra are computed separately (each unit-normalised)
and averaged. Multitapering trades a little resolution for a large variance
reduction, which matters on 2 s windows. All shape features use the band
$[f_{lo}, f_{hi}]$ = 1–40 Hz (P) or 0.7–40 Hz (S); 40 Hz stays below the
digitiser's anti-alias roll-off. Inside the band define the normalised
weights $w(f) = P(f)/\sum P$.

| column | formula | meaning |
|---|---|---|
| `centroid_hz` | $\exp\big(\sum w \ln f\big)$ | log-frequency centre of mass. Larger events put relatively more energy at low frequency (lower corner) |
| `bw_logf` | $\sqrt{\sum w(\ln f - \ln f_c)^2}$, $f_c$ = centroid | spread in log-frequency (octave-like bandwidth) |
| `flatness` | $\exp(\overline{\ln P}) / \overline{P}$ (geometric/arithmetic mean) | 1 = white, → 0 = peaked. Noise-dominated spectra are flat — leak-prone |
| `entropy` | $-\sum w \ln w / \ln N_f$ | normalised spectral entropy, 0–1 |
| `slope` | slope of least squares $\ln P$ vs $\ln f$ | overall tilt; above the corner a Brune source gives velocity power $\propto f^{-2}$ |
| `peak_hz` | $\arg\max_f P(f)$ | dominant frequency of velocity |
| `rolloff50_hz`, `rolloff85_hz`, `rolloff95_hz` | smallest $f$ with $\sum_{f' \le f} w \ge q$ | frequencies below which 50/85/95% of the energy lies |
| `frac_1_2` … `frac_32_45` | $\sum_{f\in[a,b)} P / \sum P$ | energy fraction per octave band: 1–2, 2–4, 4–8, 8–16, 16–32, 32–45 Hz (S also includes 0.7–1 Hz in the total) |

## 2. Source-model fits — `f_pz_*`, `f_sh_*`

Fitted on $\ln D(f)$ linearly interpolated to 40 log-spaced frequencies in
the band, so the many high-frequency FFT bins do not dominate. The level
term $\ln\Omega_0$ is a free nuisance parameter and is discarded — it is the
seismic moment (amplitude), the thing we are hiding.

**Brune ω² + attenuation** (Brune 1970; Anderson & Hough 1984):

$$\ln D(f) = \ln\Omega_0 - \ln\!\Big(1 + \big(f/f_c\big)^2\Big) - \pi\kappa f$$

Grid over $f_c$ (80 log steps, 0.3–45 Hz); for each $f_c$, $(\ln\Omega_0, \kappa)$
by linear least squares; keep the lowest residual.

| column | meaning |
|---|---|
| `brune_fc_hz` | corner frequency. Constant stress drop gives $f_c \propto M_0^{-1/3}$, i.e. $\log f_c \approx -0.5\,M_w$ + const. **The main expected magnitude carrier** |
| `brune_kappa` | high-frequency decay κ (s). Mostly site and path, though a magnitude trend has been reported (Marmara) |
| `brune_rms` | RMS residual in $\ln D$, fit quality |

**Boatwright, free fall-off** (Boatwright 1980, γ = 1 form):

$$\ln D(f) = \ln\Omega_0 - \ln\!\Big(1 + \big(f/f_c\big)^{n}\Big)$$

Grid over $f_c$ (as above) × $n \in [1, 4]$ (31 steps); $\ln\Omega_0$ is the mean
residual.

| column | meaning |
|---|---|
| `boat_fc_hz` | corner frequency without κ (absorbs attenuation into $n$) |
| `boat_n` | high-frequency fall-off exponent (2 for ω²) |
| `boat_rms` | fit quality |
| `fc_at_edge` | 1 if either $f_c$ is stuck at the grid edge (0.3 or 45 Hz), meaning the corner was not resolved |

**Non-parametric corner** (Andrews 1986; Snoke 1987) — no source shape assumed:

$$f_c^{\text{Snoke}} = \frac{1}{2\pi}\sqrt{\frac{\int P(f)\,df}{\int D(f)^2\,df}} = \frac{1}{2\pi}\sqrt{\frac{\int \lvert V\rvert^2 df}{\int \lvert D\rvert^2 df}}$$

over the band (trapezoid rule). For an ω² spectrum over a wide band this
returns $f_c$ exactly. Column `snoke_fc_hz`.

## 3. Attenuation-corrected source fits — `f_pz_qc_*`, `f_sh_qc_*`

At M2–3 the corner (≈10–30 Hz) sits where anelastic attenuation removes
energy, so a far station sees an apparently lower corner. Before fitting,
the power spectrum is corrected with a regional frequency-dependent Q:

$$P_{qc}(f) = P(f)\,\exp\!\Big(\frac{2\pi f\,t}{Q(f)}\Big),\qquad Q(f) = Q_0 f^{\,n}$$

with $t$ = travel time ($t_P$ or $t_S$ after origin), $Q_0^S = 60$,
$n = 0.85$, $Q_0^P = 1.5\,Q_0^S$. These sit inside published Turkish estimates
(Marmara $Q_S \approx 40 f^{1.03}$; East Anatolian Fault $Q_c \approx 57.5 f^{0.82}$;
SW Anatolia $Q_0$ 33–82, $n$ 0.79–0.91). Geometric spreading does not depend
on frequency and cancels in shape. Origin times are rounded to whole seconds,
so $t$ is a little late (≤ 1 s); the effect on the correction is small.

Columns: same as §2 with `qc_` inserted — `qc_brune_fc_hz`, `qc_brune_kappa`
(now ≈ site κ₀), `qc_brune_rms`, `qc_boat_fc_hz`, `qc_boat_n`, `qc_boat_rms`,
`qc_fc_at_edge`, `qc_snoke_fc_hz`.

## 4. P/S corner ratio — `f_fc_p_over_s_brune`, `f_fc_p_over_s_snoke`

$$\frac{f_c^{P}}{f_c^{S}}\quad\text{from the attenuation-corrected Brune or Snoke estimates}$$

Rupture models and observations give ≈ 1.2–1.7. Values far from that suggest
one of the two corners is unresolved, typically pinned by attenuation or the
band edge. Also used as a blast discriminant in the literature.

## 5. Early-warning period parameters — `f_pz_tau_c`, `f_pz_tau_p_max`

On the high-passed P window of Z.

**τc** (Kanamori 2005), effective period of the first seconds of P:

$$\tau_c = 2\pi\sqrt{\frac{\sum u^2}{\sum v^2}}$$

**τp max** (Allen & Kanamori 2003), recursive predominant period, α = 0.99:

$$X_i = \alpha X_{i-1} + v_i^2,\quad D_i = \alpha D_{i-1} + \dot v_i^2,\quad \tau_p(i) = 2\pi\sqrt{X_i/D_i}$$

$\dot v$ by central differences × $f_s$. Reported: $\max_i \tau_p(i)$ for
$i \ge 0.5$ s (earlier values are start-up transients).

Both grow with magnitude for small to moderate events and saturate around
M6.5 (outside this catalogue). Note: the window is 2 s here, not the usual 3 s.

## 6. Peak-ratio periods — `f_pz_T_dv_s`, `f_pz_T_va_s`, `f_sh_T_dv_s`, `f_sh_T_va_s`

On the high-passed P (Z) or S (horizontal vector) window, with $d$ the
displacement, $v$ velocity and $a = \dot v$; for S, peaks are of the vector norm
$\sqrt{x_N^2 + x_E^2}$:

$$T_{dv} = 2\pi\frac{\max\lvert d\rvert}{\max\lvert v\rvert},\qquad T_{va} = 2\pi\frac{\max\lvert v\rvert}{\max\lvert a\rvert}$$

For a sinusoid of period $T$ both equal $T$. They estimate the dominant period
at low and high frequency, as PGD/PGV and PGV/PGA ratios do in ground-motion
work. Larger events have longer periods.

## 7. LPDT curve — `f_lpdt_*`

Colombelli & Zollo (2015): the log of the running peak of P displacement
grows and then levels off. The time it levels off at is related to the
rupture half-duration.

$$y(t) = \log_{10}\frac{\max_{t'\le t}\lvert u(t')\rvert}{\max_{t'\le T_w}\lvert u(t')\rvert},\qquad t \ge 0.05\ \text{s}$$

Fitted with ramp + plateau $y = a + b\,\min(t, T)$, grid over $T$, $(a, b)$ by
least squares.

| column | meaning |
|---|---|
| `t_plateau_s` | $T$ + 0.05 s — break time; scales with source duration ($\propto M_0^{1/3}$) |
| `slope_dec_per_s` | $b$, initial growth rate in decades per second; small events reach their peak abruptly |
| `t90_s` | first time $y \ge \log_{10}0.9$ (displacement within 90% of its final peak) |
| `fit_rms` | ramp-plateau residual |

At M2–3 the expected source duration is ~0.05–0.2 s, near the time
resolution, so path and site broadening may dominate. Records with S−P < 0.7 s
get NaN.

## 8. Significant duration — `f_durz_*`, `f_durh_*`

Husid curve (Trifunac & Brady 1975) on band-passed Z, or N²+E², from $t_P$ to
$t_S + 10$:

$$H(t) = \frac{\int_{t_P}^{t} x^2\,dt'}{\int_{t_P}^{t_S+10} x^2\,dt'},\qquad t_q = \min\{t : H(t)\ge q\}$$

| column | formula |
|---|---|
| `d5_75_s` | $t_{0.75} - t_{0.05}$ |
| `d5_95_s` | $t_{0.95} - t_{0.05}$ |
| `d20_80_s` | $t_{0.80} - t_{0.20}$ |
| `t5_s` | $t_{0.05} - t_P$ (where energy build-up starts) |

Ground-motion duration models carry a magnitude term from source duration
plus a distance term (hence `ctl_`). Because the window is fixed, a weak
event's noise tail stretches D5-95, so D5-75 is the safer of the three.

## 9. Waveform statistics — `f_pz_*`, `f_sh_*`

On $\hat x$ (band-passed, unit peak). For `f_sh_`, computed on N and E
separately and averaged.

| column | formula | meaning |
|---|---|---|
| `kurt` | $\frac{\overline{(\hat x - \mu)^4}}{\sigma^4} - 3$ | impulsiveness |
| `abs_skew` | $\left\lvert\frac{\overline{(\hat x - \mu)^3}}{\sigma^3}\right\rvert$ | asymmetry; absolute value because the sign is polarity (radiation pattern, instrument), not size |
| `crest` | $\max\lvert\hat x\rvert / \mathrm{rms}(\hat x) = 1/\mathrm{rms}(\hat x)$ | peak-to-RMS ratio |
| `zcr` | fraction of $n$ with $\mathrm{sign}\,\hat x_{n} \ne \mathrm{sign}\,\hat x_{n-1}$ | crude dominant-frequency estimate |
| `hjorth_mob` | $\sqrt{\mathrm{var}(\Delta\hat x)/\mathrm{var}(\hat x)}$ | mean frequency (per sample) |
| `hjorth_cplx` | $\mathrm{mob}(\Delta\hat x)/\mathrm{mob}(\hat x)$ | bandwidth; 1 for a pure sine |

Hjorth *activity* ($\mathrm{var}\,x$) is amplitude and is excluded.

## 10. Complexity — `f_pz_*`, `f_sh_*`

Same windows and N/E averaging as §9.

| column | definition |
|---|---|
| `perm_ent` | Permutation entropy (Bandt & Pompe 2002), order $m=5$, delay 1: Shannon entropy of the ordinal-pattern distribution of $(x_n, \dots, x_{n+4})$, divided by $\ln 5!$. Depends only on rank order |
| `samp_ent` | Sample entropy (Richman & Moorman 2000), $m=2$, $r = 0.2\,\sigma(\hat x)$, Chebyshev distance: $-\ln(A/B)$, where $B$ and $A$ count template pairs of length $m$ and $m+1$ within $r$. NaN if $A$ or $B$ is 0 |
| `higuchi_fd` | Higuchi fractal dimension, $k_{max}=10$: $L(k) = \mathrm{mean}_m \frac{(N-1)\sum\lvert x_{m+ik} - x_{m+(i-1)k}\rvert}{\lfloor (N-m)/k\rfloor\,k^2}$; FD = $-$slope of $\ln L$ vs $\ln k$ |
| `svd_ent` | SVD entropy: normalised singular values $\bar s_i$ of the delay-embedding matrix (order 10); $-\sum \bar s_i\ln\bar s_i / \ln 10$ |

These measure how irregular the waveform is. Any magnitude signal would be
indirect, via frequency content. Lyapunov and correlation-dimension estimates
were left out because they are unreliable on windows of 200–500 samples.

## 11. Polarisation — `f_ppol_*`, `f_spol_*`

First second of the P and S windows, three band-passed components stacked
and scaled by one common peak (so the ratios between components are kept).
$\lambda_1 \ge \lambda_2 \ge \lambda_3$ are eigenvalues of the 3×3 covariance
matrix, and $\mathbf e_1$ is the principal eigenvector.

| column | formula | meaning |
|---|---|---|
| `rectilin` | $1 - \frac{\lambda_2 + \lambda_3}{2\lambda_1}$ | 1 = linear particle motion |
| `planarity` | $1 - \frac{2\lambda_3}{\lambda_1 + \lambda_2}$ | 1 = motion confined to a plane |
| `incidence_deg` | $\arccos\lvert e_{1,Z}\rvert$ | apparent incidence angle (depth, distance, near-surface velocity) |
| `log_h_over_v` | $\log_{10}\frac{\sqrt{\overline{N^2} + \overline{E^2}}}{\sqrt{\overline{Z^2}}}$ | H/V energy ratio (site amplification, wave type) |

Plus `f_log_s_over_p` $= \log_{10}\big(\mathrm{rms}_{3C}(S)/\mathrm{rms}_{3C}(P)\big)$:
S-to-P energy ratio, controlled by radiation pattern and Qp/Qs.

These are mostly path, site and mechanism controls. If one ranks high, check
it against `ctl_` before reading it as source physics.

## 12. Envelope shape — `f_envz_*`, `f_envh_*`

Envelope $e(t) = \lvert\mathcal H\{x\}\rvert$ (Hilbert), smoothed with a 0.2 s
moving average and divided by its maximum. For H:
$e_H = \sqrt{e_N^2 + e_E^2}$, renormalised. Window $[t_P - 1, t_S + 10)$;
$i_P$, $i_S$, $i_{pk}$ are the P, S and peak samples; $\Delta_{SP} = t_S - t_P$.

| column | formula | meaning |
|---|---|---|
| `t_peak_s` | $(i_{pk} - i_P)/f_s$ | time from P to the envelope peak |
| `t_peak_rel_sp` | `t_peak_s` $/\ \Delta_{SP}$ | same, as a fraction of S−P (~1 = S-wave peak) |
| `rise_10_90_s` | first $e \ge 0.9$ minus first $e \ge 0.1$ after P | rise time; grows with source duration |
| `dur_above50_s` | $\#\{e \ge 0.5\}/f_s$ | time above half peak, a pulse-width measure |
| `dur_above20_s` | $\#\{e \ge 0.2\}/f_s$ | wider version; **leak-prone** when 0.2·peak is near the noise |
| `p_over_peak` | $\max_{[i_P, i_S)} e$ | P-to-peak envelope ratio |
| `decay_per_s` | slope of least squares $\ln e$ vs $t$, from the peak to the window end | exponential decay rate of the early coda (mostly $Q$) |
| `decay_powerlaw` | slope of $\ln e$ vs $\ln t_{\text{origin}}$, same span | coda power-law exponent (spreading + $Q$) |
| `centroid_s` | $\sum e_n (n - i_P) / \sum e_n / f_s$ over $n \ge i_P$ | energy centre time after P |
| `fill` | $\overline{e}$ over $n \ge i_P$ | how full the window is (≈ duration ÷ window length) |

---

## Event medians — `f_evmed_*` (added in `train.py`)

For each event, the median over that event's records **in the same train or
test subset** of:

`pz_qc_brune_fc_hz`, `sh_qc_brune_fc_hz`, `pz_qc_snoke_fc_hz`,
`sh_qc_snoke_fc_hz`, `sh_qc_brune_kappa`, `pz_tau_c`, `pz_tau_p_max`,
`lpdt_t_plateau_s`, `durh_d5_95_s`, `sh_T_dv_s`, `fc_p_over_s_snoke`.

A single station's corner is biased by radiation pattern, directivity and
site. The network median is the standard source estimate. Only the median is
used, never the station count, because the count tracks magnitude through
recording selection. Turn off with `--no-event-medians`.

## Controls — `ctl_*` (model inputs, not features of interest)

| column | definition |
|---|---|
| `ctl_epi_km` | haversine epicentral distance, catalogue epicentre → station |
| `ctl_hypo_km` | $\sqrt{\text{epi}^2 + (\text{depth} + \text{elev}/1000)^2}$ |
| `ctl_depth_km` | catalogue depth |
| `ctl_baz_deg` | back-azimuth, station → event |
| `ctl_sp_s` | theoretical S−P (iasp91) |

They let the model separate path effects from source effects. Large events
are recorded farther away, so these predict magnitude a little by themselves:
the `ctl` baseline measures how much.

## References — `ref_*` (never model inputs)

| column | definition |
|---|---|
| `ref_log10_peak_counts` | $\log_{10}\max_n \lVert(x_Z, x_N, x_E)\rVert$ over $[t_P - 0.05,\ t_S + 10)$, band-passed counts |
| `ref_log10_snr_p` | $\log_{10}\mathrm{rms}(Z_{P})/\mathrm{rms}(Z_{noise})$, noise window $[t_P - 6, t_P - 1)$ |
| `ref_log10_snr_s` | same for N/E in the S window |
| `ref_clipped` | always 0 in the table (clipped records are dropped) |

`ref_log10_peak_counts` + `ctl_` is the "amplitude" baseline (a fitted local
magnitude relation). The SNRs drive the leak audit.

## Bibliography

- Allen & Kanamori (2003), *Science* 300 — τp max.
- Anderson & Hough (1984), *BSSA* 74 — κ.
- Andrews (1986), *AGU Monograph* 37 — integral corner frequency.
- Bandt & Pompe (2002), *PRL* 88 — permutation entropy.
- Boatwright (1980), *BSSA* 70 — spectral fall-off model.
- Brune (1970), *JGR* 75 — ω² source.
- Colombelli & Zollo (2015), *GJI* 202, 1158 — LPDT plateau.
- Higuchi (1988), *Physica D* 31 — fractal dimension.
- Kanamori (2005), *Annu. Rev. Earth Planet. Sci.* 33 — τc.
- Richman & Moorman (2000), *Am. J. Physiol.* 278 — sample entropy.
- Snoke (1987), *BSSA* 77 — corner frequency from spectral integrals.
- Trifunac & Brady (1975), *BSSA* 65 — significant duration.
- Afshari & Stewart (2016), *Earthquake Spectra* 32 — duration GMM with source-duration magnitude scaling.
- Regional Q: Marmara $Q_S \approx 40 f^{1.03}$ (*PEPI* 2004); East Anatolian Fault $Q_c \approx 57.5 f^{0.82}$ (*PAGEOPH* 2011); SW Anatolia $Q_0$ 33–82, $n$ 0.79–0.91.
