"""Streamlit-інтерфейс для перегляду результатів і підбору конфігурацій."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.recommend_config import DATASET_PATHS, SORT_COLUMNS, recommend


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"


st.set_page_config(
    page_title="Прогнозування продуктивності edge-мереж",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    # Кешування зменшує час перемикання між вкладками з великими CSV-звітами.
    return pd.read_csv(path)


@st.cache_data
def load_dataset(search_space: str) -> pd.DataFrame:
    return pd.read_csv(DATASET_PATHS[search_space])


def show_metrics() -> None:
    # Метрики зберігаються окремо для кожного search space і цільової змінної.
    st.caption(
        "Таблиця порівнює регресійні моделі для прогнозування latency та energy. "
        "Найкраща модель обирається за мінімальним MAE."
    )

    metric_files = [
        "nasbench201_latency_metrics.json",
        "nasbench201_energy_metrics.json",
        "fbnet_latency_metrics.json",
        "fbnet_energy_metrics.json",
    ]
    rows = []
    for filename in metric_files:
        path = REPORTS_DIR / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            report = json.load(file)

        for result in report["results"]:
            rows.append(
                {
                    "search_space": report["search_space"],
                    "target": report["target"],
                    "rows_used": report["rows_used"],
                    "model": result["model"],
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "mape_percent": result["mape_percent"],
                    "best": result["model"] == report["best_model"],
                }
            )

    if not rows:
        st.warning("Метрики не знайдено. Спочатку запустіть training-скрипти.")
        return

    df = pd.DataFrame(rows)
    show_only_best = st.checkbox("Показувати тільки найкращі моделі", value=False)
    if show_only_best:
        df = df[df["best"]].copy()

    df["mae"] = df["mae"].round(3)
    df["rmse"] = df["rmse"].round(3)
    df["mape_percent"] = df["mape_percent"].round(2)

    st.dataframe(df, use_container_width=True, hide_index=True)


def show_error_analysis() -> None:
    # Вкладка використовує агреговані звіти, які формує analyze_baseline_errors.py.
    device_path = REPORTS_DIR / "error_by_device.csv"
    dataset_path = REPORTS_DIR / "error_by_dataset.csv"

    if device_path.exists() and dataset_path.exists():
        device_errors = load_csv(device_path).sort_values("mae", ascending=False)
        dataset_errors = load_csv(dataset_path).sort_values("mae", ascending=False)

        highest_device = device_errors.iloc[0]
        lowest_device = device_errors.iloc[-1]
        highest_dataset = dataset_errors.iloc[0]

        summary_columns = st.columns(3)
        summary_columns[0].metric(
            "Найбільша помилка пристрою",
            highest_device["device"],
            f"MAE = {highest_device['mae']:.4f}",
        )
        summary_columns[1].metric(
            "Найменша помилка пристрою",
            lowest_device["device"],
            f"MAE = {lowest_device['mae']:.4f}",
        )
        summary_columns[2].metric(
            "Найбільша помилка dataset",
            highest_dataset["dataset"],
            f"MAE = {highest_dataset['mae']:.4f}",
        )
    else:
        device_errors = None
        dataset_errors = None

    left, right = st.columns(2)

    with left:
        st.subheader("Помилка за пристроями")
        if device_errors is not None:
            st.bar_chart(device_errors.set_index("device")["mae"])
            st.dataframe(device_errors, use_container_width=True, hide_index=True)
        else:
            st.warning("Спочатку запустіть `python src/analyze_baseline_errors.py`.")

    with right:
        st.subheader("Помилка за dataset")
        if dataset_errors is not None:
            st.bar_chart(dataset_errors.set_index("dataset")["mae"])
            st.dataframe(dataset_errors, use_container_width=True, hide_index=True)
        else:
            st.warning("Спочатку запустіть `python src/analyze_baseline_errors.py`.")


def show_feature_importance() -> None:
    # Permutation importance показує, наскільки зростає MAE після перемішування ознаки.
    path = REPORTS_DIR / "feature_importance.csv"
    if not path.exists():
        st.warning("Спочатку запустіть `python src/analyze_feature_importance.py`.")
        return

    df = load_csv(path)
    df = df.sort_values("importance_mean_mae_increase", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    plot_df = df.sort_values("importance_mean_mae_increase", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df["feature"], plot_df["importance_mean_mae_increase"])
    ax.set_xlabel("Збільшення MAE після перемішування ознаки")
    ax.set_ylabel("Ознака")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def show_split_strategies() -> None:
    path = REPORTS_DIR / "split_strategy_results.csv"
    if not path.exists():
        st.warning("Спочатку запустіть `python src/evaluate_split_strategies.py`.")
        return

    df = load_csv(path)
    st.caption(
        "`Architecture split` перевіряє узагальнення на архітектури, яких не було "
        "у train. `Device split` перевіряє узагальнення на hardware-платформи, "
        "яких не було у train."
    )

    random_row = df[df["split"] == "random"].iloc[0]
    architecture_row = df[df["split"] == "architecture"].iloc[0]
    device_rows = df[df["split"] == "device"].sort_values("mae", ascending=False)
    worst_device = device_rows.iloc[0]

    summary_columns = st.columns(3)
    summary_columns[0].metric("Random split MAE", f"{random_row['mae']:.4f}")
    summary_columns[1].metric("Architecture split MAE", f"{architecture_row['mae']:.4f}")
    summary_columns[2].metric(
        "Найгірший device split",
        worst_device["test_group"],
        f"MAE = {worst_device['mae']:.4f}",
    )

    st.dataframe(df, use_container_width=True, hide_index=True)


def show_recommendations() -> None:
    left, right = st.columns([1, 2])

    with left:
        # Значення за замовчуванням беруться з розподілу реальних benchmark-даних.
        search_space = st.selectbox("Простір пошуку", sorted(DATASET_PATHS))
        source_df = load_dataset(search_space)
        positive_latency = source_df.loc[
            source_df["latency"].notna() & (source_df["latency"] > 0),
            "latency",
        ]
        positive_energy = source_df.loc[
            source_df["energy"].notna() & (source_df["energy"] > 0),
            "energy",
        ]

        device_options = ["any"] + sorted(source_df["device"].dropna().unique().tolist())
        selected_device = st.selectbox("Пристрій", device_options)

        selected_dataset = None
        if search_space == "nasbench201":
            dataset_options = ["any"] + sorted(source_df["dataset"].dropna().unique().tolist())
            selected_dataset = st.selectbox("Dataset", dataset_options)

        latency_step = 0.01 if search_space == "fbnet" else 0.1
        energy_step = 0.01 if search_space == "fbnet" else 0.1

        max_latency = st.number_input(
            "Максимальна прогнозована latency",
            min_value=0.0,
            value=float(positive_latency.quantile(0.75)),
            step=latency_step,
            format="%.4f" if search_space == "fbnet" else "%.2f",
            help="Значення за замовчуванням відповідає 75-му перцентилю latency для вибраного search space.",
        )
        max_energy = st.number_input(
            "Максимальна прогнозована energy",
            min_value=0.0,
            value=float(positive_energy.quantile(0.75)),
            step=energy_step,
            format="%.4f" if search_space == "fbnet" else "%.2f",
            help="Значення за замовчуванням відповідає 75-му перцентилю energy для вибраного search space.",
        )
        sort_by = st.selectbox("Сортувати за", sorted(SORT_COLUMNS))
        top_n = st.slider("Кількість рекомендацій", min_value=5, max_value=50, value=10, step=5)

    args = SimpleNamespace(
        # recommend_config.recommend очікує argparse-подібний об'єкт.
        search_space=search_space,
        max_latency=max_latency,
        max_energy=max_energy,
        device=None if selected_device == "any" else selected_device,
        dataset=None if selected_dataset in (None, "any") else selected_dataset,
        sort_by=sort_by,
        top_n=top_n,
        output=None,
    )

    with right:
        try:
            recommendations = recommend(args)
        except FileNotFoundError as error:
            st.warning(str(error))
            return

        if recommendations.empty:
            st.info("Немає конфігурацій, які відповідають вибраним обмеженням.")
        else:
            st.caption(
                "Рекомендації сформовано на основі прогнозованих значень latency та "
                "energy, отриманих ML-моделями, а не на основі прямих benchmark-вимірювань."
            )

            metric_columns = st.columns(3)
            metric_columns[0].metric("Знайдено рекомендацій", len(recommendations))
            metric_columns[1].metric(
                "Найкраща predicted latency",
                f"{recommendations['predicted_latency'].min():.4f}",
            )
            metric_columns[2].metric(
                "Найкраща predicted energy",
                f"{recommendations['predicted_energy'].min():.4f}",
            )

            show_full_table = st.checkbox("Показати повну таблицю ознак", value=False)
            if show_full_table:
                table = recommendations
            elif search_space == "nasbench201":
                compact_columns = [
                    "arch_id",
                    "device",
                    "dataset",
                    "predicted_latency",
                    "predicted_energy",
                    "arch_name",
                ]
                table = recommendations[[column for column in compact_columns if column in recommendations.columns]]
            else:
                compact_columns = [
                    "block_id",
                    "device",
                    "predicted_latency",
                    "predicted_energy",
                    "block_name",
                ]
                table = recommendations[[column for column in compact_columns if column in recommendations.columns]]

            st.dataframe(table, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Прогнозування продуктивності нейронних мереж на edge-пристроях")

    tabs = st.tabs(
        [
            "Метрики",
            "Аналіз помилок",
            "Важливість ознак",
            "Стратегії split",
            "Рекомендації",
        ]
    )

    with tabs[0]:
        show_metrics()
    with tabs[1]:
        show_error_analysis()
    with tabs[2]:
        show_feature_importance()
    with tabs[3]:
        show_split_strategies()
    with tabs[4]:
        show_recommendations()


if __name__ == "__main__":
    main()
