# UrbanVerse Urban Time-Series Foundation Model PoC

Zero-shot urban traffic forecasting with **TimesFM 2.5** and **Moirai 2.0** as candidate temporal-modeling backbones for the UrbanVerse research framework.

## Research objective

This proof of concept evaluates the following UrbanVerse branch:

```text
Urban Time Series → Foundation Model → Temporal Urban Representation
```

The current PoC tests the **zero-shot forecasting behavior** of pretrained time-series foundation models on an urban traffic signal, without training or fine-tuning on METR-LA. It does not claim to implement the full UrbanVerse architecture or to expose a finalized latent city representation.

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

Both models transferred directly to the second sensor without retraining.

## Multi-window robustness evaluation

To reduce dependence on a single forecast window, 10 consecutive daily candidate windows were pre-specified from `2012-06-04` through `2012-06-13`, all starting at `13:30`.

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

The robustness result does not identify one universal winner: both pretrained models improve aggregate performance over persistence, while relative performance varies across the evaluated windows.

## Predictive uncertainty diagnostics

The executed robustness notebook also contains q10–q50–q90 diagnostics for W01 and W09.

### W01 — main evaluation window

| Model | q10–q90 empirical coverage | Mean interval width |
|---|---:|---:|
| TimesFM 2.5 | 23 / 24 = 95.8% | 15.9496 |
| Moirai 2.0 | 22 / 24 = 91.7% | 13.3498 |

### W09 — post-hoc illustrative hard window

| Model | q10–q90 empirical coverage | Mean interval width |
|---|---:|---:|
| TimesFM 2.5 | 18 / 24 = 75.0% | 28.0303 |
| Moirai 2.0 | 19 / 24 = 79.2% | 26.8294 |

W09 is described as a hard/stress case **post hoc**, after inspecting the robustness errors; the original candidate-window schedule itself was pre-specified. The reported coverage is descriptive empirical q10–q90 coverage on these selected windows, not a formal calibration or confidence guarantee.

## UrbanVerse connection

```text
Urban Time Series
      ↓
TimesFM 2.5 / Moirai 2.0
      ↓
Candidate Temporal Modeling Backbone
      ↓
Future Temporal Representation Integration
      ↓
Multi-scale Urban Encoder / Shared City State
```

The current PoC validates forecasting behavior for the temporal branch only. Integration with spatial, mobility, graph, or knowledge representations is future work and is not implemented here.

## Repository structure

```text
urbanverse-timeseries-poc/
├── README.md
├── requirements.txt
├── requirements-common.txt
├── requirements-timesfm.txt
├── requirements-moirai2.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── README.md
│   ├── 01_timesfm_inference.ipynb
│   ├── 02_moirai2_inference.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_second_sensor_zero_shot.ipynb
│   └── 05_robustness_test.ipynb
├── results/
│   ├── main_sensor/
│   ├── second_sensor/
│   ├── robustness/
│   └── uncertainty/
├── scripts/
│   └── run_moirai2_robustness.py
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_utils.py
    └── metrics.py
```

The CSV artifacts in `results/` correspond to outputs produced by the executed notebooks. The experiment plots remain preserved in the executed notebook outputs; no separately redrawn or assistant-generated scientific figures are used as replacements.

## Reproducibility notes

TimesFM and Moirai require different compatible environments in this PoC and should not be forced into one shared runtime.

- TimesFM 2.5 was validated on a Google Colab NVIDIA Tesla T4 runtime.
- Moirai 2.0 was validated in an isolated environment with `uni2ts==2.0.0` and PyTorch 2.4.1 CPU.
- Model inference is zero-shot in all reported experiments.
- Inference speed is not compared because the validated pipelines use different devices/environments.
- Raw METR-LA data and pretrained model weights are not committed to the repository.

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
- [x] Notebooks `01`–`05` organized in GitHub
- [x] Notebook-produced result CSVs organized under `results/`
- [x] Notebook-native plots preserved in executed notebook outputs
