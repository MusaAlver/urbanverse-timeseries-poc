# UrbanVerse Urban Time-Series Foundation Model PoC

Zero-shot urban traffic forecasting with **TimesFM 2.5** and **Moirai 2.0** on **METR-LA** for the temporal branch of UrbanVerse.

```text
Urban Time Series -> Foundation Model -> Temporal Urban Representation
```

This repository is a **proof of concept**, not a full forecasting paper. The final revision focuses on five questions: are the metrics/alignment correct, do the models beat simple baselines, do point forecasts actually follow the trajectory, how does behavior change across time scales, and was the original 8-hour 5-minute context too short?

## Figure provenance

All figures shown in this README are **direct outputs of the experiment/evaluation scripts** using the saved Ground Truth and model prediction files. No manually redrawn summary figure is used for the scientific results below.

## Key findings

- **Hourly is the strongest executed scale.** Against a stronger **Seasonal Naive (24h)** baseline, TimesFM has mean MAE `1.567` vs `2.985` and wins **8/9** windows. Moirai has mean MAE `2.218` and wins **4/9**.
- **The flat Persistence baseline was structurally weak at a 24-hour horizon.** Seasonal Naive reduces mean MAE from `12.563` to `2.985`; the hourly result therefore should not be presented as a simple “87% better” claim.
- **TimesFM's hourly advantage is more consistent than Moirai's.** TimesFM's largest gains occur where Seasonal Naive itself has the highest error; when the previous-day pattern is already strong, the extra gain is much smaller.
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

## 2. Multi-scale temporal dynamics

The observed series is resampled **before** inference at each scale. Five-minute forecasts are not averaged after prediction.

| Scale | Context | Horizon | Windows | Persistence MAE | TimesFM MAE | Moirai MAE |
|---|---:|---:|---:|---:|---:|---:|
| 5-minute | 8 h | 2 h | 10 | 4.225 | 3.464 | 3.453 |
| Hourly | 7 d | 24 h | 9 | 12.563 | **1.567** | 2.218 |
| Daily | 28 d | 14 d | 10 | 3.434 | **1.303** | 1.812 |
| Weekly | 8 wk | 2 wk | 1 | visual only | visual only | visual only |

Raw MAE should **not** be compared across scales as if they were the same forecasting task. The meaningful comparison is model vs baseline **within each scale**.

### 5-minute

![5-minute forecast comparison](figures/multiscale/sensor_773062/5min_forecast_comparison.png)

### Hourly with stronger baseline

**Seasonal Naive (24h)** copies each forecast hour from the same hour one day earlier.

| Method | Mean MAE | Mean RMSE | Wins vs Seasonal Naive | First-diff corr. | Directional accuracy | Variability ratio |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 12.563 | 17.274 | — | — | — | 0.000 |
| Seasonal Naive (24h) | 2.985 | 5.298 | baseline | 0.795 | 0.758 | 0.849 |
| **TimesFM 2.5** | **1.567** | **2.233** | **8/9** | **0.907** | **0.792** | **0.982** |
| Moirai 2.0 | 2.218 | 3.403 | 4/9 | 0.864 | 0.778 | 0.886 |

TimesFM has about **47.5% lower mean MAE** than Seasonal Naive. Moirai has about **25.7% lower mean MAE**, but its advantage is less consistent across windows. TimesFM's largest gains over Seasonal Naive occur in the hourly windows where the seasonal baseline itself has the highest error; when the previous-day pattern is already captured well, the additional gain is much smaller. This remains a descriptive PoC result because the 9 hourly horizons overlap and are concentrated on 25–26 June 2012.

![Hourly forecast with Seasonal Naive](figures/multiscale/sensor_773062/hourly_forecast_comparison_with_seasonal_naive.png)

### Daily

![Daily forecast comparison](figures/multiscale/sensor_773062/daily_forecast_comparison.png)

### Weekly — exploratory only

Only two future points are available in the executed weekly window, so the plot is shown for completeness rather than model ranking.

![Weekly exploratory comparison](figures/multiscale/sensor_773062/weekly_forecast_comparison.png)

## 3. Context-length control at fixed 5-minute resolution

The multi-scale comparison originally changed both resolution and historical coverage: the 5-minute experiment used 8 hours of history, while the hourly experiment used 7 days. To isolate that confound, a paired 5-minute control keeps the same forecast origins, same 24-step / 2-hour targets, and changes only the history length.

| Context | Input points | TimesFM MAE | Moirai MAE | TimesFM first-diff corr. | Moirai first-diff corr. |
|---|---:|---:|---:|---:|---:|
| 8 h | 96 | 3.385 | 3.137 | -0.128 | 0.107 |
| 24 h | 288 | 2.913 | **2.412** | 0.103 | 0.176 |
| 7 d | 2016 | **2.560** | 2.461 | **0.240** | **0.229** |

The largest improvement generally appears once at least one full daily cycle is visible. Additional history beyond 24 hours gives smaller and model-dependent gains. Longer history therefore explains **part, but not all**, of the gap between the five-minute and hourly results.

The paired quality filter leaves five forecast origins, one for each time regime, but all five are from **2012-06-27**. This limits generalization of the context-length result.

![TimesFM context-length control](figures/context_ablation/sensor_773062/timesfm_context_length_comparison.png)

![Moirai context-length control](figures/context_ablation/sensor_773062/moirai_context_length_comparison.png)

## 4. Interpretation

The PoC no longer supports a simple claim that one foundation model is universally better. Instead:

- TimesFM shows the most consistent advantage at the executed hourly scale, including against a daily Seasonal Naive baseline.
- Moirai improves mean hourly MAE over Seasonal Naive but does not beat it consistently across windows.
- At five-minute resolution, longer context improves accuracy and some movement diagnostics, but high-frequency step-to-step tracking remains weak.
- Weekly evidence is visual/exploratory only.

## 5. Limitations

- The hourly evaluation contains overlapping 24-hour horizons and is concentrated on 25–26 June 2012.
- The five-minute context-length ablation uses five regimes from a single day, 27 June 2012, after paired data-quality filtering.
- The weekly experiment contains one context and only two target points, so no strong numeric conclusion is drawn.
- This is a single-dataset, primarily single-sensor PoC; it is not evidence of broad urban forecasting generalization.
- No task-specific training or fine-tuning on METR-LA was performed.

## Reproducibility

The repository includes saved model predictions, evaluation CSVs, baseline utilities, metric checks, multi-scale/context evaluation scripts, and tests. Raw METR-LA data and pretrained model checkpoints are not stored in the normal repository.
