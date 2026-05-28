from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "hwnasbench_nasbench201.csv"
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


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_split(
    split_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_group: str | None = None,
) -> dict[str, float | int | str | None]:
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "split": split_name,
        "test_group": test_group,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mape_percent": mape(y_test, predictions),
    }


def random_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    return train_df.copy(), test_df.copy()


def architecture_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arch_ids = pd.Series(df["arch_id"].unique())
    train_arch_ids, test_arch_ids = train_test_split(
        arch_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_df = df[df["arch_id"].isin(train_arch_ids)].copy()
    test_df = df[df["arch_id"].isin(test_arch_ids)].copy()
    return train_df, test_df


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df = df[df[TARGET] > 0].copy()

    results = []

    train_df, test_df = random_split(df)
    results.append(evaluate_split("random", train_df, test_df))

    train_df, test_df = architecture_split(df)
    results.append(evaluate_split("architecture", train_df, test_df))

    for device in sorted(df["device"].unique()):
        train_df = df[df["device"] != device].copy()
        test_df = df[df["device"] == device].copy()
        results.append(evaluate_split("device", train_df, test_df, test_group=device))

    report = pd.DataFrame(results)
    output_path = REPORTS_DIR / "split_strategy_results.csv"
    report.to_csv(output_path, index=False)

    print(f"Saved split strategy results to: {output_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
