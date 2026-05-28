from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def summarize_errors(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for group_values, group in df.groupby(group_columns):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {
            column: value for column, value in zip(group_columns, group_values)
        }
        row.update(
            {
                "rows": int(len(group)),
                "mae": float(mean_absolute_error(group["actual_latency"], group["predicted_latency"])),
                "rmse": rmse(group["actual_latency"], group["predicted_latency"]),
                "mean_error": float(group["error"].mean()),
                "median_absolute_error": float(group["absolute_error"].median()),
                "max_absolute_error": float(group["absolute_error"].max()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("mae", ascending=False)


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

    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    predictions = pd.Series(model.predict(x_test), index=x_test.index, name="predicted_latency")

    error_analysis = df.loc[x_test.index, ["dataset", "arch_id", "device", "arch_str"]].copy()
    error_analysis["actual_latency"] = y_test
    error_analysis["predicted_latency"] = predictions
    error_analysis["error"] = error_analysis["predicted_latency"] - error_analysis["actual_latency"]
    error_analysis["absolute_error"] = error_analysis["error"].abs()
    error_analysis["squared_error"] = error_analysis["error"] ** 2

    error_by_device = summarize_errors(error_analysis, ["device"])
    error_by_dataset = summarize_errors(error_analysis, ["dataset"])

    error_analysis_path = REPORTS_DIR / "error_analysis.csv"
    error_by_device_path = REPORTS_DIR / "error_by_device.csv"
    error_by_dataset_path = REPORTS_DIR / "error_by_dataset.csv"

    error_analysis.to_csv(error_analysis_path, index=False)
    error_by_device.to_csv(error_by_device_path, index=False)
    error_by_dataset.to_csv(error_by_dataset_path, index=False)

    print(f"Saved row-level errors to: {error_analysis_path}")
    print(f"Saved device-level errors to: {error_by_device_path}")
    print(error_by_device.to_string(index=False))
    print(f"\nSaved dataset-level errors to: {error_by_dataset_path}")
    print(error_by_dataset.to_string(index=False))


if __name__ == "__main__":
    main()
