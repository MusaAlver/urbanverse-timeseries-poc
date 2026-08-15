# Notebooks

The executed notebooks are the primary source of truth for this PoC. Reported metrics, prediction tables, and experiment plots should be traced back to these notebook runs.

## Validated workflow

1. `01_timesfm_inference.ipynb` — TimesFM 2.5 zero-shot inference and output validation for the fixed primary-sensor experiment; also contains the validated T4 second-sensor TimesFM run.
2. `02_moirai2_inference.ipynb` — Moirai 2.0 zero-shot inference on the same primary-sensor context and forecast horizon.
3. `03_model_comparison.ipynb` — identical-window TimesFM/Moirai comparison, MAE/RMSE calculation, prediction table, and the main comparison plot.
4. `04_second_sensor_zero_shot.ipynb` — cleaned zero-shot comparison on sensor `717608`; the historical invalid shared-CPU TimesFM NaN attempt is excluded from the final comparison.
5. `05_robustness_test.ipynb` — pre-specified daily-window audit, persistence baseline, seven-window robustness evaluation, aggregate comparisons, and W01/W09 uncertainty diagnostics.

## Reproducibility rules

- Primary sensor: `773062`
- Second sensor: `717608`
- Sampling interval: 5 minutes
- Context: 96 observations
- Horizon: 24 observations
- Point forecast: q0.5 / median
- Training or fine-tuning on METR-LA: none

Raw METR-LA files and pretrained model checkpoints are not stored inside the notebooks or repository. The plots shown in the executed notebook outputs are retained as the original visual record; they are not replaced with separately redrawn scientific figures.
