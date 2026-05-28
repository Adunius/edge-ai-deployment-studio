from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "hwnasbench_nasbench201.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "nasbench201_latency_predictor.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"

CATEGORICAL_FEATURES = ["dataset", "device"]
NUMERIC_FEATURES = [
    "base_channels",
    "num_cells",
    "num_classes",
    "op_count_avg_pool_3x3",
    "op_count_nor_conv_1x1",
    "op_count_nor_conv_3x3",
    "op_count_skip_connect",
    "op_count_none",
]
TARGET = "latency"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ROWS = 5_000
N_REPEATS = 5


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train_performance.py --search-space nasbench201 --target latency first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df = df[df[TARGET] > 0].copy()

    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    x = df[feature_columns]
    y = df[TARGET]

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    if len(x_test) > MAX_ROWS:
        x_sample = x_test.sample(n=MAX_ROWS, random_state=RANDOM_STATE)
        y_sample = y_test.loc[x_sample.index]
    else:
        x_sample = x_test
        y_sample = y_test

    with MODEL_PATH.open("rb") as file:
        pipeline = pickle.load(file)

    importance = permutation_importance(
        pipeline,
        x_sample,
        y_sample,
        scoring="neg_mean_absolute_error",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    feature_types = {feature: "categorical" for feature in CATEGORICAL_FEATURES}
    feature_types.update({feature: "numeric" for feature in NUMERIC_FEATURES})

    report = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "feature_type": [feature_types[feature] for feature in x_sample.columns],
            "importance_mean_mae_increase": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    report = report.sort_values("importance_mean_mae_increase", ascending=False)

    output_path = REPORTS_DIR / "feature_importance.csv"
    report.to_csv(output_path, index=False)

    print(f"Saved permutation feature importance to: {output_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
