"""Навчання моделей прогнозування latency та energy для HW-NAS-Bench."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGETS = ("latency", "energy")


@dataclass(frozen=True)
class DatasetConfig:
    # Конфігурація задає, які колонки використовуються як ознаки для кожного search space.
    search_space: str
    dataset_path: Path
    categorical_features: list[str]
    numeric_features: list[str]


DATASET_CONFIGS = {
    "nasbench201": DatasetConfig(
        search_space="nasbench201",
        dataset_path=PROCESSED_DIR / "hwnasbench_nasbench201.csv",
        categorical_features=["dataset", "device"],
        numeric_features=[
            "base_channels",
            "num_cells",
            "num_classes",
            "op_count_avg_pool_3x3",
            "op_count_nor_conv_1x1",
            "op_count_nor_conv_3x3",
            "op_count_skip_connect",
            "op_count_none",
        ],
    ),
    "fbnet": DatasetConfig(
        search_space="fbnet",
        dataset_path=PROCESSED_DIR / "hwnasbench_fbnet.csv",
        categorical_features=["device"],
        numeric_features=[
            "input_h",
            "input_w",
            "cin",
            "cout",
            "expansion",
            "kernel",
            "stride",
            "group",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a performance predictor.")
    parser.add_argument(
        "--search-space",
        choices=sorted(DATASET_CONFIGS),
        required=True,
        help="Dataset search space to train on.",
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        required=True,
        help="Performance target to predict.",
    )
    return parser.parse_args()


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def build_pipeline(
    model: object,
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool = False,
) -> Pipeline:
    # Категоріальні ознаки кодуються One-Hot, числові передаються напряму або масштабуються.
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_candidates() -> dict[str, object]:
    # Порівнюються три рівні складності: baseline, boosting і невелика MLP-модель.
    return {
        "linear_regression": LinearRegression(),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
        "neural_network_mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=128,
            max_iter=400,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(
    model_name: str,
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    predictions = pipeline.predict(x_test)
    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mape_percent": mape(y_test, predictions),
    }


def load_training_data(
    config: DatasetConfig,
    target: str,
) -> tuple[pd.DataFrame, pd.Series, int, int]:
    df = pd.read_csv(config.dataset_path)
    rows_total = len(df)

    # Для навчання залишаються тільки записи з повним набором ознак і додатним target.
    required_columns = config.categorical_features + config.numeric_features + [target]
    df = df[required_columns].dropna().copy()
    df = df[df[target] > 0].copy()

    x = df[config.categorical_features + config.numeric_features]
    y = df[target]
    return x, y, rows_total, len(df)


def train(config: DatasetConfig, target: str) -> dict[str, object]:
    x, y, rows_total, rows_used = load_training_data(config, target)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    results = []
    trained_pipelines: dict[str, Pipeline] = {}

    # Кожна модель навчається на однаковому train/test split для коректного порівняння.
    for model_name, model in build_candidates().items():
        pipeline = build_pipeline(
            model=model,
            categorical_features=config.categorical_features,
            numeric_features=config.numeric_features,
            scale_numeric=model_name == "neural_network_mlp",
        )
        pipeline.fit(x_train, y_train)
        trained_pipelines[model_name] = pipeline
        results.append(evaluate(model_name, pipeline, x_test, y_test))

    best_result = min(results, key=lambda item: item["mae"])
    best_model_name = str(best_result["model"])
    best_pipeline = trained_pipelines[best_model_name]

    # Найкраща модель визначається за мінімальним MAE і зберігається для рекомендацій.
    metrics = {
        "search_space": config.search_space,
        "dataset": str(config.dataset_path.relative_to(PROJECT_ROOT)),
        "rows_total": rows_total,
        "rows_used": rows_used,
        "target": target,
        "features": {
            "categorical": config.categorical_features,
            "numeric": config.numeric_features,
        },
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "results": results,
        "best_model": best_model_name,
    }

    metrics_path = REPORTS_DIR / f"{config.search_space}_{target}_metrics.json"
    model_path = MODELS_DIR / f"{config.search_space}_{target}_predictor.pkl"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with model_path.open("wb") as file:
        pickle.dump(best_pipeline, file)

    return metrics


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIGS[args.search_space]
    metrics = train(config, args.target)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {REPORTS_DIR / f'{config.search_space}_{args.target}_metrics.json'}")
    print(f"Saved model to: {MODELS_DIR / f'{config.search_space}_{args.target}_predictor.pkl'}")


if __name__ == "__main__":
    main()
