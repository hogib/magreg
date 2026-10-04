# magreg

Magnitude regression from amplitude-blind waveform features (KO network,
AFAD catalogue). Features are invariant to the scale of the record and never
reference the noise level, so the model cannot simply read amplitude.

```
uv sync
uv run magreg extract     # -> data/features/features.parquet
uv run magreg select      # magnitude-balanced event list
uv run magreg audit       # which features still track SNR
uv run pytest
```

See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the design,
feature definitions, selection method and known leaks.
