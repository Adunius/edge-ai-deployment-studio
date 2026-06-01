"""Analyze LG G8X real-device validation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = EXPERIMENT_DIR / "reports"
VALIDATION_PATH = REPORTS_DIR / "real_device_validation_lg_g8x.csv"
ANALYZED_PATH = REPORTS_DIR / "real_device_validation_lg_g8x_analyzed.csv"
SUMMARY_PATH = REPORTS_DIR / "real_device_validation_lg_g8x_summary.json"


def main() -> None:
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}. Run the Android benchmark first."
        )

    df = pd.read_csv(VALIDATION_PATH)
    df = df.copy()
    df["absolute_error_ms"] = (
        df["measured_median_ms"] - df["predicted_latency"]
    ).abs()
    df["relative_error_percent"] = (
        df["absolute_error_ms"] / df["measured_median_ms"].replace(0, np.nan)
    ) * 100
    df["predicted_rank"] = df["predicted_latency"].rank(method="average")
    df["measured_rank"] = df["measured_median_ms"].rank(method="average")
    df["rank_difference"] = (df["predicted_rank"] - df["measured_rank"]).abs()

    grouped = (
        df.groupby("latency_group", observed=True)
        .agg(
            rows=("block_id", "count"),
            predicted_median_ms=("predicted_latency", "median"),
            measured_median_ms=("measured_median_ms", "median"),
            absolute_error_median_ms=("absolute_error_ms", "median"),
        )
        .reset_index()
    )

    summary = {
        "rows": int(len(df)),
        "benchmark_device": str(df["benchmark_device"].iloc[0]),
        "real_device": str(df["real_device"].iloc[0]),
        "mae_ms": float(mean_absolute_error(df["measured_median_ms"], df["predicted_latency"])),
        "rmse_ms": float(
            np.sqrt(mean_squared_error(df["measured_median_ms"], df["predicted_latency"]))
        ),
        "median_absolute_error_ms": float(df["absolute_error_ms"].median()),
        "median_relative_error_percent": float(df["relative_error_percent"].median()),
        "pearson_correlation": float(
            df["predicted_latency"].corr(df["measured_median_ms"], method="pearson")
        ),
        "spearman_rank_correlation": float(
            df["predicted_latency"].corr(df["measured_median_ms"], method="spearman")
        ),
        "latency_group_summary": grouped.to_dict(orient="records"),
        "note": (
            "LG G8X ThinQ is not the original HW-NAS-Bench pixel3 device. "
            "This validation checks practical ranking consistency on a real Android device."
        ),
    }

    df.to_csv(ANALYZED_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved analyzed validation rows to: {ANALYZED_PATH}")
    print(f"Saved validation summary to: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
