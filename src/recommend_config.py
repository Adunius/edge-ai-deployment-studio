from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATASET_PATHS = {
    "nasbench201": PROCESSED_DIR / "hwnasbench_nasbench201.csv",
    "fbnet": PROCESSED_DIR / "hwnasbench_fbnet.csv",
}

FEATURES = {
    "nasbench201": {
        "categorical": ["dataset", "device"],
        "numeric": [
            "base_channels",
            "num_cells",
            "num_classes",
            "op_count_avg_pool_3x3",
            "op_count_nor_conv_1x1",
            "op_count_nor_conv_3x3",
            "op_count_skip_connect",
            "op_count_none",
        ],
        "output": [
            "search_space",
            "dataset",
            "arch_id",
            "device",
            "predicted_latency",
            "predicted_energy",
            "arch_name",
            "base_channels",
            "num_cells",
            "num_classes",
            "op_count_avg_pool_3x3",
            "op_count_nor_conv_1x1",
            "op_count_nor_conv_3x3",
            "op_count_skip_connect",
            "op_count_none",
            "arch_str",
        ],
    },
    "fbnet": {
        "categorical": ["device"],
        "numeric": [
            "input_h",
            "input_w",
            "cin",
            "cout",
            "expansion",
            "kernel",
            "stride",
            "group",
        ],
        "output": [
            "search_space",
            "block_id",
            "device",
            "predicted_latency",
            "predicted_energy",
            "input_h",
            "input_w",
            "cin",
            "cout",
            "expansion",
            "kernel",
            "stride",
            "group",
            "block_name",
        ],
    },
}

SORT_COLUMNS = {
    "latency": ["predicted_latency", "predicted_energy"],
    "energy": ["predicted_energy", "predicted_latency"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recommend deployment configurations using trained predictors."
    )
    parser.add_argument(
        "--search-space",
        choices=sorted(DATASET_PATHS),
        required=True,
        help="Search space to recommend from.",
    )
    parser.add_argument(
        "--max-latency",
        type=float,
        default=None,
        help="Maximum allowed predicted latency.",
    )
    parser.add_argument(
        "--max-energy",
        type=float,
        default=None,
        help="Maximum allowed predicted energy.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device filter.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional NASBench201 dataset filter.",
    )
    parser.add_argument(
        "--sort-by",
        choices=sorted(SORT_COLUMNS),
        default="latency",
        help="Primary ranking objective.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of recommendations to return.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path.",
    )
    return parser.parse_args()


def model_path(search_space: str, target: str) -> Path:
    return MODELS_DIR / f"{search_space}_{target}_predictor.pkl"


def load_model(search_space: str, target: str) -> object:
    path = model_path(search_space, target)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run src/train_performance.py "
            f"--search-space {search_space} --target {target} first."
        )
    with path.open("rb") as file:
        return pickle.load(file)


def prepare_candidates(search_space: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATHS[search_space])
    feature_columns = FEATURES[search_space]["categorical"] + FEATURES[search_space]["numeric"]
    return df.dropna(subset=feature_columns).copy()


def add_predictions(df: pd.DataFrame, search_space: str) -> pd.DataFrame:
    feature_columns = FEATURES[search_space]["categorical"] + FEATURES[search_space]["numeric"]
    candidates = df.copy()

    latency_model = load_model(search_space, "latency")
    candidates["predicted_latency"] = latency_model.predict(candidates[feature_columns])

    energy_model = load_model(search_space, "energy")
    candidates["predicted_energy"] = energy_model.predict(candidates[feature_columns])

    return candidates


def filter_candidates(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    candidates = df.copy()
    candidates = candidates[
        (candidates["predicted_latency"] > 0)
        & (candidates["predicted_energy"] > 0)
    ]

    if args.max_latency is not None:
        candidates = candidates[candidates["predicted_latency"] <= args.max_latency]

    if args.max_energy is not None:
        candidates = candidates[candidates["predicted_energy"] <= args.max_energy]

    if args.device is not None:
        candidates = candidates[candidates["device"] == args.device]

    if args.dataset is not None:
        if "dataset" not in candidates.columns:
            raise ValueError("--dataset is only available for nasbench201.")
        candidates = candidates[candidates["dataset"] == args.dataset]

    return candidates


def recommend(args: argparse.Namespace) -> pd.DataFrame:
    candidates = prepare_candidates(args.search_space)
    candidates = add_predictions(candidates, args.search_space)
    candidates = filter_candidates(candidates, args)
    tie_breaker = "arch_id" if args.search_space == "nasbench201" else "block_id"
    candidates = candidates.sort_values(
        SORT_COLUMNS[args.sort_by] + [tie_breaker],
        na_position="last",
    )

    output_columns = [
        column for column in FEATURES[args.search_space]["output"] if column in candidates.columns
    ]
    return candidates[output_columns].head(args.top_n)


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return Path(args.output)

    suffix = [
        args.search_space,
        f"sort_{args.sort_by}",
    ]
    if args.device:
        suffix.append(f"device_{args.device}")
    if args.dataset:
        suffix.append(f"dataset_{args.dataset}")
    return REPORTS_DIR / f"recommendations_{'_'.join(suffix)}.csv"


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    recommendations = recommend(args)
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_path, index=False)

    if recommendations.empty:
        print("No recommendations found for the selected constraints.")
    else:
        print(recommendations.to_string(index=False))
    print(f"\nSaved recommendations to: {output_path}")


if __name__ == "__main__":
    main()
