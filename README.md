# UrbanVerse Urban Time-Series Foundation Model PoC

Zero-shot urban time-series forecasting with **TimesFM** and **Moirai** as the temporal representation component of the UrbanVerse research framework.

## Project scope

This repository implements the **Urban Time Series → Foundation Model → Temporal Representation** branch of UrbanVerse.

The current proof of concept intentionally stays small and controlled:

- one urban variable: traffic speed
- one primary urban time series from METR-LA
- pretrained TimesFM and Moirai
- no training or fine-tuning on METR-LA
- identical context window and forecast horizon for both models
- MAE and RMSE evaluation
- ground-truth vs. model prediction visualization
- optional zero-shot transfer to a second sensor without retraining

Large multivariate and cross-city experiments are outside the scope of this PoC.

## Experimental protocol

| Item | Setting |
|---|---|
| Dataset | METR-LA |
| Urban variable | Traffic speed |
| Sampling interval | 5 minutes |
| Primary sensor | `773062` |
| Generalization sensor | `717608` |
| Context length | 96 observations (8 hours) |
| Forecast horizon | 24 observations (2 hours) |
| TimesFM | Pretrained, zero-shot |
| Moirai | Pretrained, zero-shot |
| Required metrics | MAE, RMSE |
| Training / fine-tuning | None |

Sensor selection is based on data-quality and temporal-pattern criteria **before** model performance is inspected, to avoid cherry-picking.

## UrbanVerse connection

```text
Urban Time Series
      ↓
TimesFM / Moirai
      ↓
Temporal Representation
      ↓
Multi-scale Urban Encoder
      ↓
Shared Latent City State
      ↓
Urban World Model
```

The goal of this branch is not only to generate a traffic forecast. It evaluates whether pretrained time-series foundation models can provide transferable temporal information about urban dynamics that can later contribute to hourly, daily, and longer-term components of the UrbanVerse representation.

## Repository structure

```text
urbanverse-timeseries-poc/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   └── metrics.py
├── results/
└── figures/
```

Model-specific runner modules and final notebooks will be added only after the corresponding inference pipelines are validated in the working environment.

## Current status

- [x] Research scope defined
- [x] METR-LA dataset obtained and inspected
- [x] Primary and generalization sensors selected using pre-model data criteria
- [x] Colab GPU environment verified (NVIDIA Tesla T4)
- [ ] TimesFM inference validated
- [ ] Moirai inference validated
- [ ] Same-window model comparison completed
- [ ] MAE / RMSE results generated
- [ ] Prediction visualization generated
- [ ] Second-sensor zero-shot test completed
- [ ] Final reproducibility instructions completed
