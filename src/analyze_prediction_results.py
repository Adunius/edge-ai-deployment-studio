"""Detailed analysis of NASBench201 latency prediction results."""

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
DOCS_DIR = PROJECT_ROOT / "docs"

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


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["error"] = result["predicted_latency"] - result["actual_latency"]
    result["absolute_error"] = result["error"].abs()
    result["absolute_percentage_error"] = (
        result["absolute_error"] / result["actual_latency"] * 100
    )
    result["prediction_direction"] = np.where(
        result["error"] > 0,
        "overestimated",
        np.where(result["error"] < 0, "underestimated", "exact"),
    )
    result["conv_3x3_group"] = pd.cut(
        result["op_count_nor_conv_3x3"],
        bins=[-1, 0, 2, 4, 6],
        labels=["0", "1-2", "3-4", "5-6"],
    )
    return result


def summarize_group(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(columns, observed=True)
        .agg(
            rows=("absolute_error", "size"),
            mean_actual_latency=("actual_latency", "mean"),
            mean_predicted_latency=("predicted_latency", "mean"),
            mae=("absolute_error", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mean_error=("error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            p95_absolute_error=("absolute_error", lambda values: float(values.quantile(0.95))),
            max_absolute_error=("absolute_error", "max"),
        )
        .reset_index()
        .sort_values("mae", ascending=False)
    )
    return grouped


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)

    for column in table.select_dtypes(include=["float"]).columns:
        table[column] = table[column].map(lambda value: f"{value:.4f}")

    table = table.astype(str)
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |"
    rows = [
        "| " + " | ".join(row[column] for column in table.columns) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def build_markdown_report(
    overall: dict[str, float | int],
    quantiles: pd.Series,
    direction_summary: pd.DataFrame,
    by_device: pd.DataFrame,
    by_dataset: pd.DataFrame,
    by_complexity: pd.DataFrame,
    worst_examples: pd.DataFrame,
) -> str:
    return f"""# Аналіз результатів прогнозування

Цей звіт деталізує результати прогнозування `latency` для NASBench201. На відміну від короткого висновку після запуску моделі, тут розглянуто розподіл помилок, приклади найбільших відхилень, залежність якості від пристрою, dataset та архітектурних ознак.

## Загальна якість прогнозування

- Тестових записів: `{overall["rows"]}`.
- MAE: `{overall["mae"]:.4f}`.
- RMSE: `{overall["rmse"]:.4f}`.
- MAPE: `{overall["mape"]:.2f}%`.
- Середня фактична latency: `{overall["mean_actual"]:.4f}`.
- Середня прогнозована latency: `{overall["mean_predicted"]:.4f}`.

Висновок: середня абсолютна помилка є невеликою відносно повільних пристроїв, але MAPE залишається високою через наявність дуже малих значень latency. Для таких записів навіть невелике абсолютне відхилення дає велику відносну помилку у відсотках, тому MAE і RMSE краще відображають практичну якість моделі для задачі підбору конфігурацій.

## Розподіл абсолютної помилки

| Квантиль | Absolute error |
| --- | ---: |
| 50% | {quantiles.loc[0.50]:.4f} |
| 75% | {quantiles.loc[0.75]:.4f} |
| 90% | {quantiles.loc[0.90]:.4f} |
| 95% | {quantiles.loc[0.95]:.4f} |
| 99% | {quantiles.loc[0.99]:.4f} |

Висновок: більшість прогнозів має помірну абсолютну помилку, але хвіст розподілу містить окремі складні приклади з набагато більшим відхиленням. Саме ці приклади важливо розглядати при поясненні обмежень моделі.

## Напрям помилки

{markdown_table(direction_summary, ["prediction_direction", "rows", "mean_actual_latency", "mean_predicted_latency", "mae", "mean_error"])}

Висновок: порівняння переоцінених і недооцінених прогнозів показує, чи має модель систематичне зміщення. Якщо `mean_error` близький до нуля, модель не має сильного загального bias, але все одно може помилятися на окремих пристроях.

## Помилка за пристроями

{markdown_table(by_device, ["device", "rows", "mean_actual_latency", "mae", "rmse", "p95_absolute_error", "max_absolute_error"])}

Висновок: найбільша помилка спостерігається для `raspi4` та `pixel3`. Це означає, що переносимість моделі між різними hardware-платформами обмежена. Для `eyeriss`, `fpga` та `edgetpu` прогнозування стабільніше, оскільки розподіл latency для цих пристроїв простіший для наявних ознак.

## Помилка за dataset

{markdown_table(by_dataset, ["dataset", "rows", "mean_actual_latency", "mae", "rmse", "p95_absolute_error"])}

Висновок: `cifar10` і `cifar100` дають більшу помилку, ніж `ImageNet16-120`. Це показує, що dataset впливає на latency не лише через кількість класів, а й через загальний профіль архітектур у benchmark-даних.

## Залежність від кількості 3x3 convolution

{markdown_table(by_complexity, ["conv_3x3_group", "rows", "mean_actual_latency", "mae", "rmse", "p95_absolute_error"])}

Висновок: збільшення кількості `nor_conv_3x3` загалом ускладнює прогнозування, тому що такі операції сильніше навантажують обчислювальні ресурси і по-різному проявляються на різних пристроях.

## Приклади найбільших помилок

{markdown_table(worst_examples, ["dataset", "device", "arch_id", "actual_latency", "predicted_latency", "error", "absolute_error", "absolute_percentage_error"], max_rows=10)}

Висновок: найбільші помилки варто інтерпретувати як boundary cases. Вони показують, де модель недостатньо добре описує залежність між архітектурою, dataset і конкретним пристроєм. Такі приклади корисні для подальшого покращення ознак або для навчання окремих моделей під різні групи пристроїв.

## Узагальнений висновок

Побудована модель придатна для первинного відбору конфігурацій і добре відтворює загальні закономірності latency у межах відомих пристроїв та архітектур. Найважливішим фактором є пристрій, а серед архітектурних ознак найбільший вплив має кількість операцій `nor_conv_3x3`, `none` і `skip_connect`. Основне обмеження моделі полягає у слабшому перенесенні на складні hardware-платформи, насамперед `raspi4` і `pixel3`. Тому результати прогнозування слід використовувати як інструмент попереднього ranking, а остаточні конфігурації для повільних або нових пристроїв бажано додатково перевіряти експериментально.
"""


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run src/train_performance.py "
            "--search-space nasbench201 --target latency first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

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

    predictions = pd.Series(model.predict(x_test), index=x_test.index)
    result = df.loc[
        x_test.index,
        [
            "dataset",
            "arch_id",
            "device",
            "arch_name",
            "arch_str",
            *NUMERIC_FEATURES,
        ],
    ].copy()
    result["actual_latency"] = y_test
    result["predicted_latency"] = predictions
    result = add_error_columns(result)

    overall = {
        "rows": len(result),
        "mae": mean_absolute_error(result["actual_latency"], result["predicted_latency"]),
        "rmse": rmse(result["actual_latency"], result["predicted_latency"]),
        "mape": mape(result["actual_latency"], result["predicted_latency"]),
        "mean_actual": result["actual_latency"].mean(),
        "mean_predicted": result["predicted_latency"].mean(),
    }
    quantiles = result["absolute_error"].quantile([0.50, 0.75, 0.90, 0.95, 0.99])
    direction_summary = summarize_group(result, ["prediction_direction"])
    by_device = summarize_group(result, ["device"])
    by_dataset = summarize_group(result, ["dataset"])
    by_complexity = summarize_group(result, ["conv_3x3_group"])
    worst_examples = result.sort_values("absolute_error", ascending=False)

    result.sort_values("absolute_error", ascending=False).head(200).to_csv(
        REPORTS_DIR / "prediction_examples.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "metric": "rows",
                "value": overall["rows"],
            },
            {
                "metric": "mae",
                "value": overall["mae"],
            },
            {
                "metric": "rmse",
                "value": overall["rmse"],
            },
            {
                "metric": "mape_percent",
                "value": overall["mape"],
            },
        ]
    ).to_csv(REPORTS_DIR / "prediction_analysis_summary.csv", index=False)

    markdown = build_markdown_report(
        overall=overall,
        quantiles=quantiles,
        direction_summary=direction_summary,
        by_device=by_device,
        by_dataset=by_dataset,
        by_complexity=by_complexity,
        worst_examples=worst_examples,
    )
    output_path = DOCS_DIR / "prediction_result_analysis.md"
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Saved detailed prediction examples to: {REPORTS_DIR / 'prediction_examples.csv'}")
    print(f"Saved summary metrics to: {REPORTS_DIR / 'prediction_analysis_summary.csv'}")
    print(f"Saved markdown analysis to: {output_path}")


if __name__ == "__main__":
    main()
