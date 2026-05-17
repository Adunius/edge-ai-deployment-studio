from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "hwnasbench_nasbench201.csv"
MODELS_DIR = PROJECT_ROOT / "models"
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


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def build_pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_scaled_pipeline(model: object) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate(model_name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    predictions = pipeline.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": rmse,
        "mape_percent": mape(y_test, predictions),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df = df[df[TARGET] > 0].copy()

    x = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=42,
        ),
        "neural_network_mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=512,
            max_iter=120,
            early_stopping=True,
            random_state=42,
        ),
    }

    results = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in candidates.items():
        if model_name == "neural_network_mlp":
            pipeline = build_scaled_pipeline(model)
        else:
            pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        trained_pipelines[model_name] = pipeline
        results.append(evaluate(model_name, pipeline, x_test, y_test))

    best_result = min(results, key=lambda item: item["mae"])
    best_model_name = str(best_result["model"])
    best_pipeline = trained_pipelines[best_model_name]

    metrics = {
        "dataset": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "rows_total": int(len(pd.read_csv(DATASET_PATH))),
        "rows_used": int(len(df)),
        "target": TARGET,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "test_size": 0.2,
        "results": results,
        "best_model": best_model_name,
    }

    metrics_path = REPORTS_DIR / "latency_baseline_metrics.json"
    model_path = MODELS_DIR / "latency_predictor_baseline.pkl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with model_path.open("wb") as file:
        pickle.dump(best_pipeline, file)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved best model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
