"""Analyze real-device validation results and derived profiling features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEVICE_FILES = {
    "lg_g8x_thinq": [
        REPORTS_DIR / "real_device_validation_lg_g8x_thinq.csv",
        REPORTS_DIR / "real_device_validation_lg_g8x.csv",
    ],
    "redmi_note_9_pro": [
        REPORTS_DIR / "real_device_validation_redmi_note_9_pro.csv",
    ],
}
DEVICE_INFO_PATH = REPORTS_DIR / "device_info.csv"
DATASET_PATH = REPORTS_DIR / "real_android_fbnet_dataset.csv"
ANALYZED_PATH = REPORTS_DIR / "real_android_fbnet_dataset_analyzed.csv"
SUMMARY_PATH = REPORTS_DIR / "real_android_fbnet_dataset_summary.json"
PREDICTION_METRICS_PATH = REPORTS_DIR / "real_android_fbnet_prediction_metrics.csv"


def q25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def q75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def q90(values: pd.Series) -> float:
    return float(values.quantile(0.90))


def add_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add approximate convolution profiling features for the generated proxy blocks."""
    df = df.copy()

    input_h = df["input_h"].astype(int)
    input_w = df["input_w"].astype(int)
    cin = df["cin"].astype(int)
    cout = df["cout"].astype(int)
    expansion = df["expansion"].astype(int)
    kernel = df["kernel"].astype(int)
    stride = df["stride"].astype(int)
    group = df["group"].astype(int)

    expanded_channels = cin * expansion
    output_h = np.ceil(input_h / stride).astype(int)
    output_w = np.ceil(input_w / stride).astype(int)
    spatial_positions = output_h * output_w

    expand_macs = np.where(
        expansion > 1,
        input_h * input_w * cin * expanded_channels,
        0,
    )
    expand_params = np.where(
        expansion > 1,
        cin * expanded_channels,
        0,
    )

    effective_groups = np.where(
        (expanded_channels % group == 0) & (cout % group == 0),
        group,
        1,
    )
    spatial_macs = (
        spatial_positions
        * kernel
        * kernel
        * (expanded_channels / effective_groups)
        * cout
    )
    spatial_params = kernel * kernel * (expanded_channels / effective_groups) * cout

    estimated_macs = expand_macs + spatial_macs
    estimated_params = expand_params + spatial_params

    df["output_h"] = output_h
    df["output_w"] = output_w
    df["expanded_channels"] = expanded_channels
    df["effective_groups"] = effective_groups.astype(int)
    df["expand_macs"] = expand_macs.astype(np.int64)
    df["spatial_macs"] = spatial_macs.astype(np.int64)
    df["estimated_macs"] = estimated_macs.astype(np.int64)
    df["estimated_flops"] = (2 * estimated_macs).astype(np.int64)
    df["estimated_params"] = estimated_params.astype(np.int64)
    df["macs_per_input_pixel"] = estimated_macs / (input_h * input_w)
    df["measured_tops"] = df["estimated_flops"] / (
        df["measured_median_ms"].replace(0, np.nan) / 1000
    ) / 1e12
    df["cpu_tops"] = df["estimated_flops"] / (
        df["cpu_per_run_ms"].replace(0, np.nan) / 1000
    ) / 1e12
    return df


def evaluate_prediction_models(df: pd.DataFrame) -> list[dict[str, float | str | int]]:
    feature_columns = [
        "predicted_latency",
        "input_h",
        "input_w",
        "cin",
        "cout",
        "expansion",
        "kernel",
        "stride",
        "group",
        "output_h",
        "output_w",
        "expanded_channels",
        "effective_groups",
        "estimated_macs",
        "estimated_flops",
        "estimated_params",
        "macs_per_input_pixel",
    ]
    target_columns = {
        "latency_median_ms": "measured_median_ms",
        "cpu_per_run_ms": "cpu_per_run_ms",
        "pss_after_kb": "pss_after_kb",
        "native_heap_after_kb": "native_heap_after_kb",
    }

    available_features = [column for column in feature_columns if column in df.columns]
    x = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    groups = df["block_id"]
    n_splits = min(6, int(groups.nunique()))
    splitter = GroupKFold(n_splits=n_splits)

    metrics = []
    for metric_name, target_column in target_columns.items():
        if target_column not in df.columns:
            continue

        y = df[target_column].replace([np.inf, -np.inf], np.nan)
        valid = y.notna()
        x_valid = x.loc[valid]
        y_valid = y.loc[valid]
        groups_valid = groups.loc[valid]
        predictions = pd.Series(index=y_valid.index, dtype=float)

        for train_idx, test_idx in splitter.split(x_valid, y_valid, groups_valid):
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                random_state=42,
            )
            model.fit(x_valid.iloc[train_idx], y_valid.iloc[train_idx])
            predictions.iloc[test_idx] = model.predict(x_valid.iloc[test_idx])

        mae = mean_absolute_error(y_valid, predictions)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))
        mape = (
            (np.abs(y_valid - predictions) / y_valid.replace(0, np.nan)).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            * 100
        ).mean()
        metrics.append(
            {
                "target": metric_name,
                "rows": int(len(y_valid)),
                "model": "RandomForestRegressor",
                "validation": f"GroupKFold by block_id, n_splits={n_splits}",
                "mae": float(mae),
                "rmse": float(rmse),
                "mape_percent": float(mape),
                "r2": float(r2_score(y_valid, predictions)),
            }
        )

    return metrics


def main() -> None:
    frames = []
    for device_id, paths in DEVICE_FILES.items():
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            print(f"Skipping missing validation file: {paths[0]}")
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
    df = add_profile_features(df)
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
        "measured_mean_ms",
        "measured_median_ms",
        "measured_std_ms",
        "measured_q25_ms",
        "measured_q75_ms",
        "measured_q90_ms",
        "measured_min_ms",
        "measured_max_ms",
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
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                    q25,
                    q75,
                    q90,
                    "min",
                    "max",
                ]
            )
            .to_dict()
        )

    profile_columns = [
        "estimated_macs",
        "estimated_flops",
        "estimated_params",
        "measured_tops",
        "cpu_tops",
    ]
    profile_summary = (
        df[profile_columns]
        .agg(
                [
                    "mean",
                    "std",
                    "median",
                    q25,
                    q75,
                    q90,
                    "min",
                    "max",
                ]
            )
        .to_dict()
    )
    prediction_metrics = evaluate_prediction_models(df)
    pd.DataFrame(prediction_metrics).to_csv(PREDICTION_METRICS_PATH, index=False)

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
        "profile_feature_columns": profile_columns,
        "profile_feature_summary": profile_summary,
        "prediction_model_metrics": prediction_metrics,
        "profiling_references": [
            {
                "title": "PyTorch Profiler recipe",
                "url": "https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html",
                "used_for": (
                    "Terminology for operator execution time, call counts, input shapes, "
                    "and memory consumption in neural-network profiling."
                ),
            },
            {
                "title": "TensorFlow Profiler guide",
                "url": "https://www.tensorflow.org/guide/profiler",
                "used_for": (
                    "General profiling workflow for TensorFlow models and hardware "
                    "performance analysis."
                ),
            },
        ],
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
    print(f"Saved prediction model metrics to: {PREDICTION_METRICS_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
