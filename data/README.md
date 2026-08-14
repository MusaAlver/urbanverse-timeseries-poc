# Data

This project uses the METR-LA traffic benchmark as the primary urban time-series source.

Raw benchmark files are **not committed** to this repository. Place the local HDF5 file at:

```text
data/metr-la.h5
```

The expected structure is a pandas HDF5 DataFrame with timestamps as rows and traffic sensors as columns.

Current PoC configuration:

- primary sensor: `773062`
- zero-shot generalization sensor: `717608`
- sampling interval: 5 minutes
- context: 96 observations
- forecast horizon: 24 observations

Any preprocessing decisions used in the final experiment will be implemented in code and documented explicitly to keep the workflow reproducible.
