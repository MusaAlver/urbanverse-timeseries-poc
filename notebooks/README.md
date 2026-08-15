# Notebooks

The executed notebooks are the source of the reported PoC results.

## Core PoC workflow

1. `01_timesfm_inference.ipynb` — TimesFM 2.5 zero-shot inference for the primary traffic series; also contains the validated TimesFM run for the second sensor.
2. `02_moirai2_inference.ipynb` — Moirai 2.0 zero-shot inference on the same primary-sensor context and forecast horizon.
3. `03_model_comparison.ipynb` — same-window TimesFM/Moirai comparison, MAE/RMSE calculation, prediction table, and main comparison plot.
4. `04_second_sensor_zero_shot.ipynb` — zero-shot comparison on sensor `717608` using validated model outputs, with no retraining.

## Optional follow-up

`05_robustness_test.ipynb` contains additional robustness and uncertainty analyses created after the core PoC. It is kept as supplementary work and is not required for the main TimesFM-vs-Moirai task.

## Reproducibility settings

- Primary sensor: `773062`
- Second sensor: `717608`
- Sampling interval: 5 minutes
- Context: 96 observations
- Horizon: 24 observations
- Point forecast: q0.5 / median
- Training or fine-tuning on METR-LA: none

The second-sensor comparison in notebook `04` uses validated notebook-produced prediction artifacts and recomputes the final MAE/RMSE comparison.

Raw METR-LA files and pretrained model checkpoints are not stored in the repository.
