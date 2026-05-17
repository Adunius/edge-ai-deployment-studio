from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PICKLE_PATH = PROJECT_ROOT / "data" / "raw" / "HW-NAS-Bench-v1_0.pickle"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OP_NAMES = ("avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3", "skip_connect", "none")
FBNET_PATTERN = re.compile(
    r"ConvBlock_H(?P<input_h>\d+)_W(?P<input_w>\d+)_Cin(?P<cin>\d+)"
    r"_Cout(?P<cout>\d+)_exp(?P<expansion>\d+)_kernel(?P<kernel>\d+)"
    r"_stride(?P<stride>\d+)_group(?P<group>\d+)"
)


def load_dataset() -> dict[str, Any]:
    if not PICKLE_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {PICKLE_PATH}")

    with PICKLE_PATH.open("rb") as file:
        return pickle.load(file)


def get_device(metric_name: str, suffix: str) -> str:
    return metric_name.removesuffix(suffix)


def extract_nasbench201_features(config: dict[str, Any]) -> dict[str, Any]:
    arch_str = config["arch_str"]
    op_counts = Counter(re.findall(r"([a-z0-9_]+)~\d+", arch_str))

    features = {
        "arch_name": config.get("name"),
        "base_channels": config.get("C"),
        "num_cells": config.get("N"),
        "num_classes": config.get("num_classes"),
        "arch_str": arch_str,
    }
    for op_name in OP_NAMES:
        features[f"op_count_{op_name}"] = op_counts.get(op_name, 0)

    return features


def export_nasbench201(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dataset_name, dataset_data in data["nasbench201"].items():
        configs = dataset_data["config"]
        latency_metrics = [name for name in dataset_data if name.endswith("_latency")]

        for arch_id, config in enumerate(configs):
            architecture_features = extract_nasbench201_features(config)

            for latency_metric in latency_metrics:
                device = get_device(latency_metric, "_latency")
                energy_metric = f"{device}_energy"
                arithmetic_metric = f"{device}_arithmetic_intensity"

                row = {
                    "search_space": "nasbench201",
                    "dataset": dataset_name,
                    "arch_id": arch_id,
                    "device": device,
                    "latency": float(dataset_data[latency_metric][arch_id]),
                    "energy": (
                        float(dataset_data[energy_metric][arch_id])
                        if energy_metric in dataset_data
                        else None
                    ),
                    "arithmetic_intensity": (
                        float(dataset_data[arithmetic_metric][arch_id])
                        if arithmetic_metric in dataset_data
                        else None
                    ),
                }
                row.update(architecture_features)
                rows.append(row)

    return pd.DataFrame(rows)


def parse_fbnet_block(block_name: str) -> dict[str, Any]:
    match = FBNET_PATTERN.fullmatch(block_name)
    if not match:
        return {"block_name": block_name}

    features = {"block_name": block_name}
    features.update({key: int(value) for key, value in match.groupdict().items()})
    return features


def export_fbnet(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fbnet_data = data["fbnet"]
    latency_metrics = [name for name in fbnet_data if name.endswith("_latency")]

    for latency_metric in latency_metrics:
        device = get_device(latency_metric, "_latency")
        energy_metric = f"{device}_energy"

        for block_id, (block_name, latency) in enumerate(fbnet_data[latency_metric].items()):
            row = {
                "search_space": "fbnet",
                "block_id": block_id,
                "device": device,
                "latency": float(latency),
                "energy": (
                    float(fbnet_data[energy_metric][block_name])
                    if energy_metric in fbnet_data
                    else None
                ),
            }
            row.update(parse_fbnet_block(block_name))
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()

    nasbench201 = export_nasbench201(data)
    fbnet = export_fbnet(data)

    nasbench201_path = PROCESSED_DIR / "hwnasbench_nasbench201.csv"
    fbnet_path = PROCESSED_DIR / "hwnasbench_fbnet.csv"

    nasbench201.to_csv(nasbench201_path, index=False)
    fbnet.to_csv(fbnet_path, index=False)

    print(f"Exported {len(nasbench201):,} NASBench201 rows to {nasbench201_path}")
    print(f"Exported {len(fbnet):,} FBNet rows to {fbnet_path}")
    print("\nNASBench201 columns:")
    print(list(nasbench201.columns))
    print("\nFBNet columns:")
    print(list(fbnet.columns))


if __name__ == "__main__":
    main()
