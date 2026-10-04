# magreg

Magnitude regression from amplitude-blind waveform features (KO network,
AFAD catalogue). Features are invariant to the scale of the record and never
reference the noise level, so the model cannot simply read amplitude.

```
uv sync
uv run magreg extract     # -> data/features/features.parquet
uv run magreg select      # magnitude-balanced event list
uv run magreg audit       # which features still track SNR
uv run magreg train       # LightGBM + baselines + importance -> runs/ (leaky features excluded)
uv run pytest
```

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the design,
[docs/FEATURES.md](docs/FEATURES.md) for every feature with formulas, and
[docs/RESULTS.md](docs/RESULTS.md) for results and the known leaks.
