# Paper outline

**Working title:** Amplitude-Independent Waveform Descriptors of Earthquake
Size: A Leakage-Controlled Gradient-Boosting Study on Turkish Broadband Data

**Abstract** (~200 words): question, data, method (amplitude-invariant
features, leakage audit, doubly disjoint validation), main result
(attenuation-corrected corner frequency dominates; shape vs distance and
amplitude baselines), limitation (large-event underestimation).

1. **Introduction**: why amplitude-free size information matters (source
   physics, early warning, saturation, site/gain errors); research question;
   contributions.
2. **Background and preliminary literature review**
   2.1 Magnitude scales and source scaling (Mw, ML–Mw at small magnitudes, Brune, self-similarity)
   2.2 Spectral source parameters and attenuation (κ, Boatwright, Snoke/Andrews, regional Q in Turkey)
   2.3 Period and duration parameters from early warning (τp, τc, LPDT, significant duration)
   2.4 Machine-learning magnitude estimation
   2.5 Leakage and structured validation
   2.6 Gap addressed by this study
3. **Data**: AFAD catalogue, KOERI KO broadband waveforms, station metadata; magnitude distribution.
4. **Methodology**
   4.1 Pre-processing, travel times and P picking (iasp91/TauP, AIC, origin-time rounding)
   4.2 Windows and quality control
   4.3 Amplitude-invariance principle and the within-event leakage audit
   4.4 Feature families (table + key equations); event medians; controls
   4.5 Magnitude-balanced event selection (cap, space–time round-robin, cell-share limit)
   4.6 Model, baselines, validation design, weighting
   4.7 Importance analysis (SHAP; permutation by family, quantity, concept)
5. **Results** (leakage-filtered configuration only)
   5.1 Dataset after QC and selection
   5.2 Predictive skill vs baselines
   5.3 Magnitude-dependent bias
   5.4 Feature importance
   5.5 Corner-frequency scaling with magnitude
   5.6 Sensitivity experiments (tail weighting, ML-only)
6. **Discussion and limitations**
7. **Conclusions and future work**
8. **Declaration of AI use**

**References** (verified by search; to be checked against the originals)

**Appendix A**: complete feature list.

Figures: (1) magnitude histogram before/after selection; (2) predicted vs
catalogue magnitude; (3) bias by magnitude bin; (4) concept/quantity
importance; (5) event-median corner frequency vs magnitude.
