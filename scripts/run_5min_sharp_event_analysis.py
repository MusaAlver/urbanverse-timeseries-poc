#!/usr/bin/env python3
"""Context-calibrated sharp-event audit for executed 5-minute forecasts.

The event threshold is calibrated only from the observed pre-forecast contexts.
Forecast-period Ground Truth and model error are never used to choose the
threshold. The q90 threshold is the primary descriptive audit; q95 is saved as
a stricter sensitivity check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abrupt_change import (
    calibrate_quantile_threshold,
    step_diagnostics,
    summarize_event_conditioned,
)
from src.data_utils import prepare_context_target

MODELS = {
    "TimesFM 2.5": "timesfm_2_5",
    "Moirai 2.0": "moirai_2_0",
}


def _load_contexts(
    series_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    series_frame = pd.read_csv(series_path, parse_dates=["timestamp"])
    series = pd.Series(
        series_frame["traffic_speed"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(series_frame["timestamp"]),
        name="traffic_speed",
    )

    manifest = pd.read_csv(manifest_path, parse_dates=["forecast_origin"])
    manifest = manifest.loc[manifest["scale"] == "5min"].copy()
    if manifest.empty:
        raise ValueError("No 5-minute windows found in evaluation manifest.")

    context_rows: list[pd.DataFrame] = []
    histories: dict[str, pd.Series] = {}
    for row in manifest.itertuples(index=False):
        matches = np.flatnonzero(series.index == row.forecast_origin)
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely locate forecast origin for {row.window_id}.")
        end_position = int(matches[0])
        history, _, _ = prepare_context_target(
            series,
            end_position=end_position,
            context=int(row.context_points),
            horizon=int(row.horizon_points),
        )
        histories[row.window_id] = history
        context_rows.append(
            pd.DataFrame(
                {
                    "window_id": row.window_id,
                    "timestamp": history.index,
                    "traffic_speed": history.to_numpy(dtype=float),
                }
            )
        )

    return pd.concat(context_rows, ignore_index=True), histories


def _build_step_table(
    comparison: pd.DataFrame,
    histories: dict[str, pd.Series],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    comparison = comparison.copy()
    comparison["timestamp"] = pd.to_datetime(comparison["timestamp"])

    for scheme, threshold in thresholds.items():
        for window_id, group in comparison.groupby("window_id", sort=False):
            if window_id not in histories:
                raise ValueError(f"Missing prepared context for {window_id}.")
            group = group.sort_values("timestamp")
            last_context_value = float(histories[window_id].iloc[-1])
            for model, column in MODELS.items():
                step = step_diagnostics(
                    timestamps=group["timestamp"],
                    y_true=group["ground_truth"],
                    y_pred=group[column],
                    last_context_value=last_context_value,
                    threshold=threshold,
                    window_id=window_id,
                    model=model,
                )
                step.insert(0, "threshold_scheme", scheme)
                step.insert(1, "threshold_value", threshold)
                frames.append(step)

    return pd.concat(frames, ignore_index=True)


def _build_event_details_q90(step: pd.DataFrame) -> pd.DataFrame:
    abrupt = step[(step["threshold_scheme"] == "q90") & step["abrupt"]].copy()
    base = abrupt[abrupt["model"] == "TimesFM 2.5"][
        [
            "window_id",
            "timestamp",
            "threshold_value",
            "ground_truth",
            "actual_delta",
            "abs_actual_delta",
        ]
    ].copy()
    base["event_direction"] = np.where(base["actual_delta"] > 0, "rise", "drop")

    for model, prefix in [("TimesFM 2.5", "timesfm"), ("Moirai 2.0", "moirai")]:
        model_rows = abrupt[abrupt["model"] == model][
            [
                "window_id",
                "timestamp",
                "prediction",
                "predicted_delta",
                "absolute_error",
                "amplitude_ratio",
                "direction_correct",
            ]
        ].copy()
        model_rows = model_rows.rename(
            columns={
                "prediction": f"{prefix}_prediction",
                "predicted_delta": f"{prefix}_predicted_delta",
                "absolute_error": f"{prefix}_absolute_error",
                "amplitude_ratio": f"{prefix}_amplitude_ratio",
                "direction_correct": f"{prefix}_direction_correct",
            }
        )
        base = base.merge(model_rows, on=["window_id", "timestamp"], how="left")

    return base.sort_values("abs_actual_delta", ascending=False).reset_index(drop=True)


def _plot_q90(details: pd.DataFrame, output_path: Path) -> None:
    gt = details["actual_delta"].to_numpy(dtype=float)
    tf = details["timesfm_predicted_delta"].to_numpy(dtype=float)
    mo = details["moirai_predicted_delta"].to_numpy(dtype=float)
    threshold = float(details["threshold_value"].iloc[0])

    bound = float(max(np.max(np.abs(gt)), np.max(np.abs(tf)), np.max(np.abs(mo))))
    lim = bound + max(1.0, 0.08 * bound)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(gt, tf, label="TimesFM 2.5", alpha=0.85)
    ax.scatter(gt, mo, label="Moirai 2.0", alpha=0.85, marker="x")
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        linestyle="--",
        linewidth=1.0,
        label="Ideal: forecast change = Ground Truth change",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Ground Truth one-step change (5 min)")
    ax.set_ylabel("Forecast one-step change (5 min)")
    ax.set_title(
        "Context-calibrated sharp 5-minute event response\n"
        f"q90 context threshold: |Δ| ≥ {threshold:.3f}"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results/multiscale/sensor_773062/5min_forecast_comparison_all_windows.csv"),
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=Path("results/multiscale/sensor_773062/5min_series.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/multiscale/sensor_773062/multiscale_evaluation_manifest.csv"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/sharp_events/sensor_773062"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/sharp_events/sensor_773062/5min_sharp_event_response.png"),
    )
    args = parser.parse_args()

    context, histories = _load_contexts(args.series, args.manifest)
    thresholds = {
        "q90": calibrate_quantile_threshold(context, quantile=0.90),
        "q95": calibrate_quantile_threshold(context, quantile=0.95),
    }
    comparison = pd.read_csv(args.comparison, parse_dates=["timestamp"])
    step = _build_step_table(comparison, histories, thresholds)

    summaries: list[pd.DataFrame] = []
    for scheme, threshold in thresholds.items():
        subset = step[step["threshold_scheme"] == scheme]
        summary = summarize_event_conditioned(subset)
        summary.insert(0, "threshold_scheme", scheme)
        summary.insert(1, "threshold_value", threshold)
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    details = _build_event_details_q90(step)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    step.to_csv(args.results_dir / "5min_sharp_event_step_level.csv", index=False)
    summary.to_csv(args.results_dir / "5min_sharp_event_summary.csv", index=False)
    details.to_csv(args.results_dir / "5min_sharp_event_details_q90.csv", index=False)

    protocol = {
        "sensor_id": "773062",
        "scale": "5min",
        "windows": int(comparison["window_id"].nunique()),
        "threshold_calibration": "pooled absolute one-step changes from prepared pre-forecast contexts only",
        "q90_threshold": thresholds["q90"],
        "q95_threshold": thresholds["q95"],
        "q90_future_events": int(len(details)),
        "q90_future_event_windows": int(details["window_id"].nunique()),
        "selection_uses_forecast_ground_truth": False,
        "selection_uses_model_error": False,
        "primary_descriptive_threshold": "q90",
        "sensitivity_threshold": "q95",
    }
    (args.results_dir / "5min_sharp_event_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    _plot_q90(details, args.figure)

    print(json.dumps(protocol, indent=2))
    print("\nEvent-conditioned summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
