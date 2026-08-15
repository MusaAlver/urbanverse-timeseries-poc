
import time
import numpy as np
import pandas as pd
import torch

from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# ============================================================
# MOIRAI 2.0 — ROBUSTNESS TEST
# 7 PRE-SPECIFIED CLEAN WINDOWS
# ============================================================

MODEL_ID = "Salesforce/moirai-2.0-R-small"
WINDOWS_PATH = "/content/robustness_windows_sensor_773062.csv"

PRED_PATH = "/content/robustness_moirai2_predictions.csv"
METRICS_PATH = "/content/robustness_moirai2_metrics.csv"
AUDIT_PATH = "/content/robustness_moirai2_output_audit.csv"
SUMMARY_PATH = "/content/robustness_moirai2_summary.csv"

EXPECTED_WINDOWS = [
    "W01", "W03", "W04", "W05", "W08", "W09", "W10"
]

CONTEXT_LENGTH = 96
FORECAST_HORIZON = 24

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. LOAD LOCKED WINDOWS
# ============================================================

df = pd.read_csv(
    WINDOWS_PATH,
    parse_dates=["timestamp"]
)

actual_windows = sorted(df["window_id"].unique().tolist())

assert actual_windows == EXPECTED_WINDOWS
assert len(df) == 7 * 120

print("Locked robustness dataset: OK")
print("Windows:", actual_windows)

# ============================================================
# 2. LOAD MOIRAI 2.0 ONCE
# ============================================================

print("\nLoading:", MODEL_ID)

module = Moirai2Module.from_pretrained(
    MODEL_ID
)

model = Moirai2Forecast(
    module=module,
    prediction_length=FORECAST_HORIZON,
    context_length=CONTEXT_LENGTH,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)

model = model.to("cpu")
model.eval()

quantile_levels = list(module.quantile_levels)

assert len(quantile_levels) == 9
assert 0.5 in quantile_levels

median_idx = quantile_levels.index(0.5)

print("Moirai 2.0 model: OK")
print("Quantiles:", quantile_levels)
print("q=0.5 index:", median_idx)

# ============================================================
# 3. RUN ALL 7 WINDOWS
# ============================================================

prediction_rows = []
metric_rows = []
audit_rows = []

for window_id in EXPECTED_WINDOWS:

    window = df[df["window_id"] == window_id].copy()

    context_df = (
        window[window["split"] == "context"]
        .sort_values("timestamp")
    )

    gt_df = (
        window[window["split"] == "ground_truth"]
        .sort_values("timestamp")
    )

    assert len(context_df) == CONTEXT_LENGTH
    assert len(gt_df) == FORECAST_HORIZON

    context = context_df["traffic_speed"].to_numpy(
        dtype=np.float32
    )

    y_true = gt_df["traffic_speed"].to_numpy(
        dtype=np.float64
    )

    assert context.shape == (96,)
    assert y_true.shape == (24,)

    assert np.isfinite(context).all()
    assert np.isfinite(y_true).all()

    # --------------------------------------------------------
    # ZERO-SHOT FORECAST
    # --------------------------------------------------------

    start = time.perf_counter()

    with torch.no_grad():
        forecast = model.predict(
            [context]
        )

    elapsed = time.perf_counter() - start

    forecast = np.asarray(forecast)

    shape_ok = (
        forecast.shape == (1, 9, 24)
    )

    if shape_ok:
        prediction = forecast[
            0,
            median_idx,
            :
        ].astype(np.float64)
    else:
        prediction = np.full(
            FORECAST_HORIZON,
            np.nan
        )

    prediction_finite = bool(
        np.isfinite(prediction).all()
    )

    valid_output = (
        shape_ok
        and prediction.shape == (24,)
        and prediction_finite
    )

    # IMPORTANT:
    # Any failed clean window remains in the audit.
    # No replacement / no performance-based exclusion.
    if valid_output:

        mae = np.mean(
            np.abs(y_true - prediction)
        )

        rmse = np.sqrt(
            np.mean(
                (y_true - prediction) ** 2
            )
        )

    else:
        mae = np.nan
        rmse = np.nan

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    audit_rows.append({
        "window_id": window_id,
        "forecast_shape": str(forecast.shape),
        "shape_ok": shape_ok,
        "prediction_finite": prediction_finite,
        "valid_output": valid_output,
        "inference_seconds": elapsed,
    })

    metric_rows.append({
        "window_id": window_id,
        "model": "Moirai",
        "version": "2.0",
        "mae": mae,
        "rmse": rmse,
        "valid_output": valid_output,
    })

    for timestamp, actual, pred in zip(
        gt_df["timestamp"],
        y_true,
        prediction,
    ):
        prediction_rows.append({
            "window_id": window_id,
            "timestamp": timestamp,
            "ground_truth": actual,
            "moirai_2_0": pred,
        })

    if valid_output:
        print(
            f"{window_id}: PASS | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f}"
        )
    else:
        print(
            f"{window_id}: FAIL — invalid Moirai output"
        )

# ============================================================
# 4. SAVE RAW RESULTS
# ============================================================

predictions = pd.DataFrame(prediction_rows)
metrics = pd.DataFrame(metric_rows)
audit = pd.DataFrame(audit_rows)

predictions.to_csv(
    PRED_PATH,
    index=False
)

metrics.to_csv(
    METRICS_PATH,
    index=False
)

audit.to_csv(
    AUDIT_PATH,
    index=False
)

# ============================================================
# 5. OUTPUT AUDIT
# ============================================================

print("\n======================================")
print("MOIRAI 2.0 OUTPUT AUDIT")
print("======================================")

print(
    audit[
        [
            "window_id",
            "forecast_shape",
            "shape_ok",
            "prediction_finite",
            "valid_output",
        ]
    ].to_string(index=False)
)

valid_count = int(
    audit["valid_output"].sum()
)

print()
print(
    "Valid Moirai windows:",
    valid_count,
    "/ 7"
)

# ============================================================
# 6. STRICT COMPLETION CHECK
# ============================================================

if valid_count != 7:

    print()
    print(
        "❌ MOIRAI DID NOT PRODUCE VALID OUTPUT "
        "FOR ALL LOCKED WINDOWS"
    )

    print(
        "No window will be removed or replaced."
    )

    print(
        "Audit saved:",
        AUDIT_PATH
    )

    raise RuntimeError(
        f"Moirai valid outputs: {valid_count}/7"
    )

# ============================================================
# 7. AGGREGATE ONLY AFTER ALL 7 PASS
# ============================================================

summary = pd.DataFrame([{
    "model": "Moirai",
    "version": "2.0",
    "n_windows": len(metrics),

    "mean_mae": metrics["mae"].mean(),
    "std_mae": metrics["mae"].std(ddof=1),
    "median_mae": metrics["mae"].median(),

    "mean_rmse": metrics["rmse"].mean(),
    "std_rmse": metrics["rmse"].std(ddof=1),
    "median_rmse": metrics["rmse"].median(),
}])

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

print()
print("======================================")
print("MOIRAI 2.0 ROBUSTNESS SUMMARY")
print("======================================")

print()
print(
    metrics[
        ["window_id", "mae", "rmse"]
    ].to_string(index=False)
)

print()
print("--------------------------------------")
print("AGGREGATE")
print("--------------------------------------")

print(
    summary.to_string(index=False)
)

print()
print("Saved:")
print(PRED_PATH)
print(METRICS_PATH)
print(AUDIT_PATH)
print(SUMMARY_PATH)

print()
print("✅ ALL 7 MOIRAI ROBUSTNESS WINDOWS PASSED")
