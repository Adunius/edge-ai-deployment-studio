"""Select FBNet blocks for real-device validation on an Android phone."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "hwnasbench_fbnet.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fbnet_latency_predictor.pkl"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

FEATURE_COLUMNS = [
    "device",
    "input_h",
    "input_w",
    "cin",
    "cout",
    "expansion",
    "kernel",
    "stride",
    "group",
]
OUTPUT_COLUMNS = [
    "block_id",
    "benchmark_device",
    "real_device",
    "predicted_latency",
    "latency_group",
    "input_h",
    "input_w",
    "cin",
    "cout",
    "expansion",
    "kernel",
    "stride",
    "group",
    "block_name",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select FBNet blocks for real-device validation."
    )
    parser.add_argument(
        "--benchmark-device",
        default="pixel3",
        help="HW-NAS-Bench device used for model-side predictions.",
    )
    parser.add_argument(
        "--real-device",
        default="lg_g8x_thinq",
        help="Name of the physical device used for validation.",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=8,
        help="Number of blocks to sample from each latency group.",
    )
    parser.add_argument(
        "--output",
        default=str(REPORTS_DIR / "real_device_candidates_lg_g8x.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def load_latency_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train_performance.py "
            "--search-space fbnet --target latency first."
        )
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def prepare_candidates(benchmark_device: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["device"] == benchmark_device].copy()
    if df.empty:
        available = sorted(pd.read_csv(DATASET_PATH)["device"].dropna().unique())
        raise ValueError(
            f"No rows found for benchmark device {benchmark_device!r}. "
            f"Available devices: {available}"
        )
    return df.dropna(subset=FEATURE_COLUMNS).copy()


def assign_latency_groups(df: pd.DataFrame) -> pd.DataFrame:
    candidates = df.sort_values(["predicted_latency", "block_id"]).copy()
    labels = ["fast", "medium", "slow"]
    candidates["latency_group"] = pd.qcut(
        candidates["predicted_latency"],
        q=3,
        labels=labels,
        duplicates="drop",
    )
    return candidates


def sample_groups(df: pd.DataFrame, per_group: int) -> pd.DataFrame:
    samples = []
    for group_name, group_df in df.groupby("latency_group", observed=True):
        group_df = group_df.sort_values(["predicted_latency", "block_id"])
        if group_name == "fast":
            sample = group_df.head(per_group)
        elif group_name == "slow":
            sample = group_df.tail(per_group)
        else:
            middle = len(group_df) // 2
            start = max(0, middle - per_group // 2)
            sample = group_df.iloc[start : start + per_group]
        samples.append(sample)

    return pd.concat(samples, ignore_index=True).sort_values(
        ["latency_group", "predicted_latency", "block_id"]
    )


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates = prepare_candidates(args.benchmark_device)
    model = load_latency_model()
    candidates["predicted_latency"] = model.predict(candidates[FEATURE_COLUMNS])
    candidates = candidates[candidates["predicted_latency"] > 0].copy()
    candidates = assign_latency_groups(candidates)
    selected = sample_groups(candidates, args.per_group)

    selected = selected.rename(columns={"device": "benchmark_device"})
    selected["real_device"] = args.real_device
    selected = selected[OUTPUT_COLUMNS]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    print(f"Saved {len(selected)} real-device candidates to: {output_path}")
    print(
        selected[
            [
                "block_id",
                "benchmark_device",
                "real_device",
                "latency_group",
                "predicted_latency",
                "input_h",
                "input_w",
                "cin",
                "cout",
                "kernel",
                "stride",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
