# UrbanVerse Urban Time-Series Foundation Model PoC

Zero-shot urban traffic forecasting with **TimesFM 2.5** and **Moirai 2.0** as the temporal representation branch of the UrbanVerse research framework.

## Research objective

This proof of concept implements the following UrbanVerse component:

```text
Urban Time Series → Foundation Model → Temporal Urban Representation
```

The goal is to test whether pretrained time-series foundation models can extract useful temporal dynamics from an urban signal **without training or fine-tuning on METR-LA**.

This PoC intentionally remains controlled and small. It does not claim to implement the full UrbanVerse architecture, multivariate city modeling, or cross-city training.

## Experimental protocol

| Item | Setting |
|---|---|
| Dataset | METR-LA |
| Urban variable | Traffic speed |
| Sampling interval | 5 minutes |
| Primary sensor | `773062` |
| Zero-shot transfer sensor | `717608` |
| Context length | 96 observations = 8 hours |
| Forecast horizon | 24 observations = 2 hours |
| TimesFM checkpoint | `google/timesfm-2.5-200m-pytorch` |
| TimesFM Python package | `timesfm==2.0.2` |
| Moirai checkpoint | `Salesforce/moirai-2.0-R-small` |
| Moirai / Uni2TS package | `uni2ts==2.0.0` |
| Point forecast | q0.5 / median |
| Metrics | MAE, RMSE |
| Training / fine-tuning | None |

The two foundation models receive the **same context, same ground truth, and same 24-step horizon**.

## Main fixed-window result

Primary sensor `773062`:

- context: `2012-06-04 13:30` → `21:25`
- forecast: `2012-06-04 21:30` → `23:25`
- 96 context points, 24 forecast points

| Model | MAE | RMSE |
|---|---:|---:|
| TimesFM 2.5 | 2.3031 | 2.9794 |
| Moirai 2.0 | **1.6552** | **2.7551** |

On this specific fixed window, Moirai 2.0 achieves lower MAE and RMSE. This is a window-level result, not a claim of general model superiority.

## Zero-shot transfer to a second sensor

The same pretrained models were evaluated on sensor `717608` using the same date, time range, context length, and horizon, with **no retraining**.

| Model | MAE | RMSE |
|---|---:|---:|
| TimesFM 2.5 | 1.3665 | 2.5878 |
| Moirai 2.0 | **1.3389** | **2.5547** |

Both models transferred directly to the second sensor, supporting the intended zero-shot temporal-representation use case.

## Multi-window robustness evaluation

To reduce dependence on a single forecast window, 10 consecutive daily windows were pre-specified from `2012-06-04` through `2012-06-13`, all starting at `13:30`.

The data-quality rule was fixed before model inference:

- no NaN values
- no zero-valued measurements in context or forecast
- exact 5-minute timestamps
- 96 context + 24 forecast points

Seven windows passed the rule. Three were excluded only for data quality:

- `W02`: 9 zero values in context
- `W06`: 5 zero values in context
- `W07`: 26 zero values in context

No failed window was replaced based on model performance.

### Aggregate robustness results — 7 locked clean windows

| Model | Mean MAE | Std MAE | Median MAE | Mean RMSE | Std RMSE | Median RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | 4.5957 | 4.0625 | 2.2031 | 5.9012 | 4.9121 | 3.0811 |
| TimesFM 2.5 | 3.8102 | 2.6685 | 2.4355 | **4.6684** | **3.4498** | 2.9927 |
| Moirai 2.0 | **3.5748** | 2.9928 | **2.0551** | 4.7823 | 3.9097 | **2.9119** |

Relative to persistence:

- TimesFM 2.5 improves mean MAE by **17.1%** and mean RMSE by **20.9%**.
- Moirai 2.0 improves mean MAE by **22.2%** and mean RMSE by **19.0%**.

Per-window wins:

| Model | MAE wins | RMSE wins |
|---|---:|---:|
| Persistence | 2 / 7 | 1 / 7 |
| TimesFM 2.5 | 2 / 7 | 2 / 7 |
| Moirai 2.0 | 3 / 7 | 4 / 7 |

The main robustness conclusion is therefore not that one foundation model always wins. Rather, **both pretrained models improve aggregate performance over a simple persistence baseline, while their relative strengths vary across temporal regimes**.

## Predictive uncertainty analysis

For two diagnostic windows, q10–q50–q90 forecasts were reproduced from the same pretrained models and checked against the saved q0.5 forecasts.

### W01 — main evaluation window

| Model | q10–q90 empirical coverage | Mean interval width |
|---|---:|---:|
| TimesFM 2.5 | 23 / 24 = 95.8% | 15.9496 |
| Moirai 2.0 | 22 / 24 = 91.7% | 13.3498 |

### W09 — hard / stress window

| Model | q10–q90 empirical coverage | Mean interval width |
|---|---:|---:|
| TimesFM 2.5 | 18 / 24 = 75.0% | 28.0303 |
| Moirai 2.0 | 19 / 24 = 79.2% | 26.8294 |

From W01 to W09, both models substantially widen their q10–q90 intervals, yet empirical coverage decreases. This descriptive result suggests that the hard temporal regime is challenging not only for point forecasting but also for uncertainty calibration.

These coverage values are **diagnostic empirical coverage on the selected windows**, not a formal calibration guarantee.

## UrbanVerse connection

```text
Urban Time Series
      ↓
TimesFM 2.5 / Moirai 2.0
      ↓
Temporal Urban Representation
      ↓
Multi-scale Urban Encoder
      ↓
Shared Latent City State
      ↓
Urban World Model
```

The current PoC validates the temporal branch only. In the wider UrbanVerse concept, this representation can later be combined with spatial, mobility, graph, and knowledge representations to support multi-scale urban forecasting and world-model objectives.

## Repository structure

```text
urbanverse-timeseries-poc/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   └── 05_robustness_test.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    └── metrics.py
```

Additional validated notebooks, result CSVs, and figures are being organized into the repository as final reproducibility artifacts.

## Reproducibility notes

TimesFM and Moirai currently require different compatible environments in this PoC. They should not be forced into one shared dependency environment.

- TimesFM 2.5 was validated on a Google Colab NVIDIA Tesla T4 runtime.
- Moirai 2.0 was validated in an isolated environment with `uni2ts==2.0.0` and PyTorch 2.4.1 CPU.
- Model inference is zero-shot in all reported experiments.
- Inference speed is not compared because the two validated pipelines use different devices/environments.

## Status

- [x] Research scope defined
- [x] METR-LA dataset inspected
- [x] Primary sensor selected before model outputs
- [x] TimesFM 2.5 inference validated
- [x] Moirai 2.0 inference validated
- [x] Same-window comparison completed
- [x] MAE / RMSE comparison completed
- [x] Second-sensor zero-shot transfer completed
- [x] Persistence baseline added
- [x] Multi-window robustness evaluation completed
- [x] Predictive uncertainty diagnostics completed
- [x] Robustness / uncertainty reproducibility audits completed
- [ ] Remaining notebooks, figures, and result artifacts organized in GitHub
- [ ] Final project-level reproducibility instructions completed
