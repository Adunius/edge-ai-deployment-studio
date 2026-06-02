"""Analyze LG G8X real-device validation results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEVICE_FILES = {
    "lg_g8x_thinq": REPORTS_DIR / "real_device_validation_lg_g8x.csv",
    "redmi_note_9_pro": REPORTS_DIR / "real_device_validation_redmi_note_9_pro.csv",
}
DEVICE_INFO_PATH = REPORTS_DIR / "device_info.csv"
DATASET_PATH = REPORTS_DIR / "real_android_fbnet_dataset.csv"
ANALYZED_PATH = REPORTS_DIR / "real_android_fbnet_dataset_analyzed.csv"
SUMMARY_PATH = REPORTS_DIR / "real_android_fbnet_dataset_summary.json"


def main() -> None:
    frames = []
    for device_id, path in DEVICE_FILES.items():
        if not path.exists():
            print(f"Skipping missing validation file: {path}")
            continue
        frame = pd.read_csv(path)
        frame["device_id"] = device_id
        frame["real_device"] = device_id
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"No validation files found in {REPORTS_DIR}. Run the Android benchmark first."
        )

    df = pd.concat(frames, ignore_index=True)
    if DEVICE_INFO_PATH.exists():
        device_info = pd.read_csv(DEVICE_INFO_PATH)
        df = df.merge(device_info, on="device_id", how="left")

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
    df["predicted_rank_by_device"] = df.groupby("device_id")["predicted_latency"].rank(
        method="average"
    )
    df["measured_rank_by_device"] = df.groupby("device_id")["measured_median_ms"].rank(
        method="average"
    )
    df["rank_difference_by_device"] = (
        df["predicted_rank_by_device"] - df["measured_rank_by_device"]
    ).abs()

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

    device_grouped = (
        df.groupby("device_id", observed=True)
        .agg(
            rows=("block_id", "count"),
            measured_median_ms=("measured_median_ms", "median"),
            cpu_per_run_median_ms=("cpu_per_run_ms", "median"),
            pss_after_median_kb=("pss_after_kb", "median"),
            native_heap_after_median_kb=("native_heap_after_kb", "median"),
        )
        .reset_index()
    )

    device_metrics = []
    for device_id, group in df.groupby("device_id", observed=True):
        device_metrics.append(
            {
                "device_id": str(device_id),
                "rows": int(len(group)),
                "mae_ms": float(
                    mean_absolute_error(group["measured_median_ms"], group["predicted_latency"])
                ),
                "rmse_ms": float(
                    np.sqrt(
                        mean_squared_error(
                            group["measured_median_ms"],
                            group["predicted_latency"],
                        )
                    )
                ),
                "pearson_correlation": float(
                    group["predicted_latency"].corr(
                        group["measured_median_ms"],
                        method="pearson",
                    )
                ),
                "spearman_rank_correlation": float(
                    group["predicted_latency"].corr(
                        group["measured_median_ms"],
                        method="spearman",
                    )
                ),
                "median_cpu_per_run_ms": float(group["cpu_per_run_ms"].median())
                if "cpu_per_run_ms" in group.columns
                else None,
                "median_pss_after_kb": float(group["pss_after_kb"].median())
                if "pss_after_kb" in group.columns
                else None,
            }
        )

    resource_columns = [
        "cpu_per_run_ms",
        "cpu_wall_ratio",
        "pss_after_kb",
        "pss_delta_kb",
        "java_heap_after_kb",
        "java_heap_delta_kb",
        "native_heap_after_kb",
        "native_heap_delta_kb",
    ]
    available_resource_columns = [column for column in resource_columns if column in df.columns]
    resource_summary = {}
    if available_resource_columns:
        resource_summary = (
            df[available_resource_columns]
            .agg(["mean", "median", "min", "max"])
            .to_dict()
        )

    summary = {
        "rows": int(len(df)),
        "devices": sorted(df["device_id"].dropna().unique().tolist()),
        "device_count": int(df["device_id"].nunique()),
        "benchmark_device": str(df["benchmark_device"].iloc[0]),
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
        "device_metrics": device_metrics,
        "device_summary": device_grouped.to_dict(orient="records"),
        "resource_columns_available": available_resource_columns,
        "resource_summary": resource_summary,
        "latency_group_summary": grouped.to_dict(orient="records"),
        "note": (
            "The physical Android devices are not the original HW-NAS-Bench pixel3 device. "
            "This validation checks practical ranking consistency and device-side resource "
            "usage on real Android devices."
        ),
    }

    df.to_csv(DATASET_PATH, index=False)
    df.to_csv(ANALYZED_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved combined real-device dataset to: {DATASET_PATH}")
    print(f"Saved analyzed validation rows to: {ANALYZED_PATH}")
    print(f"Saved validation summary to: {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
