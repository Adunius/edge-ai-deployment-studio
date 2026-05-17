# Прогнозування продуктивності нейронних мереж на edge-пристроях

Репозиторій бакалаврського проєкту за темою:

**Метод та програмні засоби прогнозування продуктивності нейронних мереж на гетерогенних edge-пристроях з обмеженими обчислювальними ресурсами**

У проєкті використовується датасет HW-NAS-Bench для побудови pipeline машинного навчання, який прогнозує latency нейронних мереж під час inference на гетерогенних edge-пристроях.

## Поточний обсяг роботи

Поточна реалізація охоплює:

- розбір benchmark-даних HW-NAS-Bench у форматі `.pickle`;
- експорт benchmark-даних у табличний CSV-формат;
- аналіз структури датасету та цільових змінних;
- навчання та порівняння моделей прогнозування latency:
  - Linear Regression;
  - Gradient Boosting;
  - MLP neural network;
- збереження найкращої моделі прогнозування та метрик оцінювання.

Основною цільовою змінною є `latency`. Поле `energy` доступне лише для частини пристроїв, тому воно розглядається як додаткова ціль для подальших експериментів.

## Структура проєкту

```text
data/
  raw/
    HW-NAS-Bench-v1_0.pickle        # оригінальний benchmark-файл, не змінюється
  processed/
    hwnasbench_nasbench201.csv      # експортована таблиця NASBench201
    hwnasbench_fbnet.csv            # експортована таблиця FBNet

models/
  latency_predictor_baseline.pkl    # найкраща навчена модель прогнозування latency

notebooks/
  01_dataset_analysis.ipynb          # аналіз датасету
  02_model_training_validation.ipynb # тренування та валідація моделей

reports/
  latency_baseline_metrics.json     # метрики порівняння моделей
  plots/

src/
  inspect_hwnasbench.py             # виводить структуру pickle-файлу
  export_hwnasbench_csv.py          # експортує pickle-дані у CSV
  train_latency_baseline.py         # навчає та оцінює ML-моделі

requirements.txt
```

## Налаштування середовища

Створіть і активуйте Python virtual environment, після чого встановіть залежності:

```bash
pip install -r requirements.txt
```

Поточний код перевірявся з локальним virtual environment у `.venv`.

## Підготовка датасету

Оригінальний файл HW-NAS-Bench потрібно розмістити за шляхом:

```text
data/raw/HW-NAS-Bench-v1_0.pickle
```

Для перегляду структури сирого benchmark-файлу:

```bash
python src/inspect_hwnasbench.py
```

Для експорту benchmark-даних у CSV:

```bash
python src/export_hwnasbench_csv.py
```

У результаті створюються CSV-файли:

- `data/processed/hwnasbench_nasbench201.csv`
- `data/processed/hwnasbench_fbnet.csv`

## Навчання моделей

Для навчання моделей прогнозування latency:

```bash
python src/train_latency_baseline.py
```

Скрипт фільтрує невалідні записи з `latency <= 0`, навчає три моделі, оцінює їх на тестовій вибірці та зберігає:

- найкращу модель: `models/latency_predictor_baseline.pkl`;
- метрики: `reports/latency_baseline_metrics.json`.

## Поточні результати baseline-експерименту

Перший експеримент використовує `hwnasbench_nasbench201.csv` і прогнозує `latency`.

| Model | MAE, ms | RMSE, ms |
| --- | ---: | ---: |
| Linear Regression | 3.98 | 6.04 |
| Gradient Boosting | 0.87 | 2.16 |
| MLP Neural Network | 0.88 | 2.16 |

Найкращою поточною моделлю за MAE є **Gradient Boosting**.

## Примітки

- `latency` доступна для всіх експортованих записів.
- `energy` відсутня для частини пристроїв, тому її потрібно аналізувати на відфільтрованій підмножині даних.
- `MAPE` є нестабільною метрикою для цього датасету, оскільки частина значень latency дуже мала.
