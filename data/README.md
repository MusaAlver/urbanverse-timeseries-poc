# Data

This project uses the METR-LA traffic benchmark as the urban time-series source.

Raw benchmark files are **not committed** to this repository.

## Path used by the executed Colab notebooks

The validated notebook runs upload the HDF5 file into the Colab runtime and read it from:

```text
/content/metr-la.h5
```

This is the path used in notebooks `01`, `04`, and `05` when they access the raw METR-LA file.

For local code outside Colab, `data/metr-la.h5` can be used as a repository-local convention and passed explicitly to the helper in `src/data_utils.py`. The executed notebooks themselves should not be described as reading that local repository path.

The expected structure is a pandas HDF5 DataFrame with timestamps as rows and traffic sensors as columns.

## PoC configuration

- primary sensor: `773062`
- zero-shot transfer sensor: `717608`
- urban variable: traffic speed
- sampling interval: 5 minutes
- context length: 96 observations = 8 hours
- forecast horizon: 24 observations = 2 hours
- training / fine-tuning: none

## Fixed primary evaluation window

- context: `2012-06-04 13:30:00` → `2012-06-04 21:25:00`
- forecast: `2012-06-04 21:30:00` → `2012-06-04 23:25:00`

The second-sensor zero-shot experiment uses the same date/time window, context length, and forecast horizon.

## Robustness-window data-quality rule

The robustness notebook pre-specifies ten daily candidate windows from `2012-06-04` through `2012-06-13`, all starting at `13:30`. A candidate is retained only when it has exact timestamps and lengths, no NaN values, and no zero-valued measurements in either context or forecast.

Seven candidates satisfy the rule: `W01`, `W03`, `W04`, `W05`, `W08`, `W09`, and `W10`. `W02`, `W06`, and `W07` are excluded for data quality before model-performance comparison.

The exact evaluation-window and audit CSV outputs produced by the notebooks are stored under `results/`.
