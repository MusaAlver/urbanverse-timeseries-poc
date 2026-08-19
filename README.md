# UrbanVerse Urban Time-Series Foundation Model PoC

Zero-shot urban traffic forecasting with **TimesFM 2.5** and **Moirai 2.0** on **METR-LA** for the temporal branch of UrbanVerse.

```text
Urban Time Series -> Foundation Model -> Temporal Urban Representation
```

This repository is a **proof of concept**, not a full forecasting paper. The final revision focuses on five questions: are the metrics/alignment correct, do the models beat simple baselines, do point forecasts actually follow the trajectory, how does behavior change across time scales, and was the original 8-hour 5-minute context too short?

> **Figure policy:** only figures produced directly by the experiment/evaluation pipeline are used as scientific evidence. Manually redrawn summary figures are not used.

## Key findings

- **Hourly is the strongest executed scale.** Against **Seasonal Naive (24h)**, TimesFM has mean MAE `1.567` vs `2.985` and wins **8/9** windows. Moirai has mean MAE `2.218` and wins **4/9**.
- **Flat Persistence is structurally weak at a 24-hour horizon.** Seasonal Naive reduces mean MAE from `12.563` to `2.985`, so the hourly result should not be summarized only as “87% better than persistence.”
- **TimesFM's hourly advantage is more consistent than Moirai's.** Its largest gains occur where Seasonal Naive itself has the highest error; when the previous-day pattern is already captured well, the extra gain is much smaller.
- **Five-minute forecasts remain weak at step-to-step motion tracking.** Longer context helps, but even 7 days of 5-minute history does not approach the executed hourly trajectory metrics.
- **The original 8-hour context was a real confound, but not the whole explanation.** Most of the observed gain appears once at least one full daily cycle is visible; additional history beyond 24 hours gives smaller/model-dependent gains.
- **Weekly is exploratory only** because the executed weekly forecast contains just two target points.

## Setup

| Item | Setting |
|---|---|
| Dataset | METR-LA |
| Primary sensor | `773062` |
| Original resolution | 5 minutes |
| TimesFM | `google/timesfm-2.5-200m-pytorch` |
| Moirai | `Salesforce/moirai-2.0-R-small` |
| Task-specific fine-tuning | None |
| Point forecast | median / q0.5 |

## 1. Metric and alignment validation

The saved primary-window MAE/RMSE values were independently recomputed on the **raw traffic-speed scale**. Context ends at `2012-06-04 21:25`, Ground Truth begins at `21:30`, and no off-by-one error was found.

| Model | MAE | RMSE |
|---|---:|---:|
| TimesFM 2.5 | 2.3031 | 2.9794 |
| Moirai 2.0 | 1.6552 | 2.7551 |

The original single late-evening window is retained only as pilot provenance; the final interpretation relies on the multi-window experiments below.

![Original experiment output](figures/main_sensor/final_model_comparison_plot.png)

## 2. Multi-scale temporal dynamics

The observed series is resampled **before** inference at each scale. Five-minute forecasts are not averaged after prediction.

| Scale | Context | Horizon | Windows | Persistence MAE | TimesFM MAE | Moirai MAE |
|---|---:|---:|---:|---:|---:|---:|
| 5-minute | 8 h | 2 h | 10 | 4.225 | 3.464 | 3.453 |
| Hourly | 7 d | 24 h | 9 | 12.563 | **1.567** | 2.218 |
| Daily | 28 d | 14 d | 10 | 3.434 | **1.303** | 1.812 |
| Weekly | 8 wk | 2 wk | 1 | visual only | visual only | visual only |

Raw MAE should **not** be compared across scales as if they were the same forecasting task. The meaningful comparison is model vs baseline **within each scale**.

### Hourly with stronger baseline

**Seasonal Naive (24h)** copies each forecast hour from the same hour one day earlier.

| Method | Mean MAE | Mean RMSE | Wins vs Seasonal Naive | First-diff corr. | Directional accuracy | Variability ratio |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 12.563 | 17.274 | — | — | — | 0.000 |
| Seasonal Naive (24h) | 2.985 | 5.298 | baseline | 0.795 | 0.758 | 0.849 |
| **TimesFM 2.5** | **1.567** | **2.233** | **8/9** | **0.907** | **0.792** | **0.982** |
| Moirai 2.0 | 2.218 | 3.403 | 4/9 | 0.864 | 0.778 | 0.886 |

TimesFM has about **47.5% lower mean MAE** than Seasonal Naive. Moirai has about **25.7% lower mean MAE**, but its advantage is less consistent across windows. TimesFM's largest gains over Seasonal Naive occur in the hourly windows where the seasonal baseline itself has the highest error; when the previous-day pattern is already captured well, the additional gain is much smaller.

This remains a descriptive PoC result because the 9 hourly horizons overlap and are concentrated on 25–26 June 2012.

## 3. Context-length control at fixed 5-minute resolution

The forecast origin, 5-minute resolution, and 24-step / 2-hour target are held fixed while only the history length changes.

| Model | Context | Mean MAE | First-diff corr. | Directional accuracy | Variability ratio |
|---|---:|---:|---:|---:|---:|
| TimesFM | 8 h | 3.385 | -0.128 | 0.504 | 0.058 |
| TimesFM | 24 h | 2.913 | 0.103 | **0.558** | **0.503** |
| TimesFM | 7 d | **2.560** | **0.240** | 0.549 | 0.302 |
| Moirai | 8 h | 3.137 | 0.107 | 0.496 | 0.423 |
| Moirai | 24 h | **2.412** | 0.176 | **0.540** | **0.509** |
| Moirai | 7 d | 2.461 | **0.229** | 0.531 | 0.396 |

For TimesFM, mean MAE falls by about **24.4%** from 8 hours to 7 days. For Moirai, the best MAE occurs at **24 hours**; 7 days does not improve it further. The largest improvement generally appears once at least one full daily cycle is visible, but longer history does not fully solve high-frequency step-to-step tracking.

The paired quality filter leaves **5 origins**, one from each clock-time regime, but all five are from **2012-06-27**. This control is therefore descriptive and single-day.

## Interpretation

The final PoC does not support a blanket “foundation models are better” statement. A more defensible result is:

> **TimesFM shows consistent added value at the executed hourly scale, including against a strong previous-day seasonal baseline. Moirai's hourly advantage is more window-dependent. At 5-minute resolution, longer historical context improves error and some shape diagnostics, but does not fully solve high-frequency trajectory tracking.**

The remaining 5-minute vs hourly gap is consistent with a role for temporal aggregation/resolution, but this experiment does not isolate that factor causally.

## Reproducibility

- Saved predictions and evaluation outputs are committed under `results/`.
- Scientific evaluation utilities are under `src/`.
- Experiment/evaluation scripts are under `scripts/`.
- Raw `metr-la.h5` and pretrained checkpoints are intentionally not committed.

## Limitations

- Primary final experiments use one METR-LA sensor (`773062`).
- Five-minute and hourly executed windows are weekday-only.
- Hourly horizons overlap and are concentrated on two days.
- The paired context ablation uses five regimes from a single day after data-quality filtering.
- Weekly results are visual/exploratory because only two future points are available.
- Results are descriptive; no statistical-significance/generalization claim is made.
