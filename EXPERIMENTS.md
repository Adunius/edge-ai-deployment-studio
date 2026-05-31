# Журнал експериментів

Цей файл фіксує результати досліджень у людському форматі. Машинні звіти зберігаються в `reports/`.

## 1. NASBench201 latency baseline

**Скрипт:** `src/train_performance.py --search-space nasbench201 --target latency`

**Дані:** `data/processed/hwnasbench_nasbench201.csv`

**Задача:** прогнозування `latency` для NASBench201.

**Використано рядків:**

- всього: `281250`;
- після фільтрації `latency > 0`: `267153`.

**Ознаки:**

- категоріальні: `dataset`, `device`;
- числові: `base_channels`, `num_cells`, `num_classes`, `op_count_avg_pool_3x3`, `op_count_nor_conv_1x1`, `op_count_nor_conv_3x3`, `op_count_skip_connect`, `op_count_none`.

| Model | MAE | RMSE | MAPE, % |
| --- | ---: | ---: | ---: |
| Linear Regression | 3.979 | 6.043 | 455.166 |
| Gradient Boosting | 0.875 | 2.159 | 187.250 |
| MLP Neural Network | 0.870 | 2.157 | 185.889 |

**Висновок:** після збільшення `max_iter` для MLP найкраща модель за MAE - `MLP Neural Network`. Gradient Boosting має дуже близький результат і залишається стабільним сильним baseline для табличних даних.

## 2. Error analysis для NASBench201 baseline

**Скрипт:** `src/analyze_baseline_errors.py`

**Файли результатів:**

- `reports/error_analysis.csv`;
- `reports/error_by_device.csv`;
- `reports/error_by_dataset.csv`.

### Помилка за пристроями

| Device | Rows | MAE | RMSE |
| --- | ---: | ---: | ---: |
| raspi4 | 9216 | 2.751 | 4.609 |
| pixel3 | 9409 | 1.301 | 2.174 |
| edgegpu | 9372 | 0.699 | 0.922 |
| edgetpu | 6582 | 0.121 | 0.168 |
| fpga | 9319 | 0.083 | 0.112 |
| eyeriss | 9533 | 0.081 | 0.106 |

**Висновок:** найбільша помилка спостерігається для `raspi4` і `pixel3`. Для `fpga`, `eyeriss` та `edgetpu` помилка значно нижча.

### Помилка за dataset

| Dataset | Rows | MAE | RMSE |
| --- | ---: | ---: | ---: |
| cifar100 | 17836 | 1.096 | 2.621 |
| cifar10 | 17739 | 1.081 | 2.536 |
| ImageNet16-120 | 17856 | 0.434 | 0.821 |

**Висновок:** помилка для `ImageNet16-120` нижча, ніж для `cifar10` та `cifar100`.

## 3. Feature importance для NASBench201 baseline

**Скрипт:** `src/analyze_feature_importance.py`

**Файл результатів:** `reports/feature_importance.csv`

Метод: permutation importance з метрикою `neg_mean_absolute_error`.

| Feature | Type | MAE increase |
| --- | --- | ---: |
| device | categorical | 6.021 |
| dataset | categorical | 2.471 |
| op_count_nor_conv_3x3 | numeric | 1.728 |
| op_count_none | numeric | 1.082 |
| op_count_skip_connect | numeric | 0.672 |
| num_classes | numeric | 0.460 |
| op_count_avg_pool_3x3 | numeric | 0.291 |
| op_count_nor_conv_1x1 | numeric | 0.279 |
| base_channels | numeric | 0.000 |
| num_cells | numeric | 0.000 |

**Висновок:** найважливіший фактор - `device`. Серед архітектурних ознак найбільший внесок має `op_count_nor_conv_3x3`. Ознаки `base_channels` і `num_cells` не впливають, оскільки в цьому датасеті вони сталі.

## 4. Split strategy evaluation

**Скрипт:** `src/evaluate_split_strategies.py`

**Файл результатів:** `reports/split_strategy_results.csv`

Модель для порівняння: `HistGradientBoostingRegressor`.

| Split | Test group | Train rows | Test rows | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| random | - | 213722 | 53431 | 0.875 | 2.159 |
| architecture | - | 213750 | 53403 | 0.890 | 2.246 |
| device | edgegpu | 220278 | 46875 | 2.165 | 2.681 |
| device | edgetpu | 234375 | 32778 | 3.056 | 3.569 |
| device | eyeriss | 220278 | 46875 | 1.536 | 1.990 |
| device | fpga | 220278 | 46875 | 1.429 | 1.595 |
| device | pixel3 | 220278 | 46875 | 5.618 | 7.712 |
| device | raspi4 | 220278 | 46875 | 16.577 | 21.957 |

**Висновок:** split by architecture майже не погіршує якість порівняно з random split. Це означає, що модель непогано узагальнюється на нові архітектури в межах відомих пристроїв. Split by device значно складніший: якість суттєво падає для `raspi4` і `pixel3`, що показує обмежену здатність моделі переноситися на нову hardware-платформу без прикладів цього пристрою в train.

## 5. FBNet latency baseline

**Скрипт:** `src/train_performance.py --search-space fbnet --target latency`

**Дані:** `data/processed/hwnasbench_fbnet.csv`

**Задача:** прогнозування `latency` для FBNet ConvBlock-записів.

**Використано рядків:**

- всього у FBNet CSV: `1125`;
- після фільтрації повних ConvBlock-ознак і `latency > 0`: `960`.

**Ознаки:**

- категоріальні: `device`;
- числові: `input_h`, `input_w`, `cin`, `cout`, `expansion`, `kernel`, `stride`, `group`.

| Model | MAE | RMSE | MAPE, % |
| --- | ---: | ---: | ---: |
| Linear Regression | 1.674 | 3.816 | 584.950 |
| Gradient Boosting | 0.660 | 2.149 | 96.714 |
| MLP Neural Network | 0.562 | 1.614 | 86.580 |

**Висновок:** для FBNet найкращою моделлю за MAE є `MLP Neural Network`. Gradient Boosting також значно кращий за Linear Regression, але поступається MLP.

## Поточний короткий підсумок

На цьому етапі реалізовано:

- baseline для `NASBench201 -> latency`;
- error analysis за пристроями та datasets;
- feature importance для пояснення внеску ознак;
- порівняння split-стратегій;
- baseline для `FBNet -> latency`;
- energy prediction для `NASBench201` і `FBNet`.

Головний технічний висновок: модель добре працює при random split і майже не втрачає якість на нових архітектурах, але перенесення на повністю новий пристрій є значно складнішою задачею.

## 6. Unified performance training script

**Скрипт:** `src/train_performance.py`

Додано універсальний training script з аргументами `--search-space` і `--target`.

Приклади запуску:

```bash
python src/train_performance.py --search-space nasbench201 --target latency
python src/train_performance.py --search-space nasbench201 --target energy
python src/train_performance.py --search-space fbnet --target latency
python src/train_performance.py --search-space fbnet --target energy
```

Скрипт підтримує різні набори ознак для `NASBench201` і `FBNet`, а також цілі `latency` і `energy`. Для всіх експериментів використовується однакова схема порівняння моделей:

- Linear Regression;
- HistGradientBoostingRegressor;
- MLPRegressor.

**Висновок:** навчання для двох search space і двох target-змінних тепер запускається через один інтерфейс. Це спрощує подальше додавання нових split-стратегій або додаткових моделей.

## 7. Energy prediction

**Скрипт:** `src/train_performance.py`

Energy доступна не для всіх записів, тому експерименти виконуються тільки на підмножині рядків, де `energy` не є `NaN` і `energy > 0`.

### NASBench201 energy

**Команда:**

```bash
python src/train_performance.py --search-space nasbench201 --target energy
```

**Використано рядків:**

- всього у CSV: `281250`;
- після фільтрації energy: `140625`.

| Model | MAE | RMSE | MAPE, % |
| --- | ---: | ---: | ---: |
| Linear Regression | 4.510 | 5.624 | 330.813 |
| Gradient Boosting | 1.096 | 2.355 | 10.204 |
| MLP Neural Network | 1.162 | 2.360 | 13.921 |

**Висновок:** для `NASBench201 -> energy` найкращою моделлю є `Gradient Boosting`.

### FBNet energy

**Команда:**

```bash
python src/train_performance.py --search-space fbnet --target energy
```

**Використано рядків:**

- всього у CSV: `1125`;
- після фільтрації повних ConvBlock-ознак та energy: `576`.

| Model | MAE | RMSE | MAPE, % |
| --- | ---: | ---: | ---: |
| Linear Regression | 6.467 | 8.764 | 3179.111 |
| Gradient Boosting | 2.571 | 4.979 | 430.798 |
| MLP Neural Network | 3.315 | 6.369 | 818.581 |

**Висновок:** для `FBNet -> energy` найкращою моделлю є `Gradient Boosting`, але MAPE дуже нестабільна через малі значення energy.

## 8. Recommendation module

**Скрипт:** `src/recommend_config.py`

Мета: підібрати top-N конфігурацій розгортання за обмеженнями на прогнозовані `latency` та `energy`.

Приклади запуску:

```bash
python src/recommend_config.py --search-space nasbench201 --max-latency 5 --max-energy 20 --sort-by energy --top-n 10
python src/recommend_config.py --search-space fbnet --max-latency 1 --max-energy 5 --sort-by latency --top-n 10
```

Скрипт:

- завантажує відповідний CSV;
- завантажує latency predictor;
- завантажує energy predictor;
- рахує `predicted_latency` і `predicted_energy`;
- відкидає фізично неможливі прогнози `<= 0`;
- фільтрує конфігурації за заданими constraints;
- сортує за `latency` або `energy`;
- зберігає результат у `reports/recommendations_*.csv`.

**Висновок:** цей модуль реалізує базову версію deployment optimizer без reinforcement learning. Він використовує навчені моделі для прогнозу продуктивності та повертає конфігурації, які відповідають заданим обмеженням.

## 9. Детальний аналіз результатів прогнозування

**Скрипт:** `src/analyze_prediction_results.py`

**Файли результатів:**

- `docs/prediction_result_analysis.md`;
- `reports/prediction_examples.csv`;
- `reports/prediction_analysis_summary.csv`.

Цей етап додано для поглибленої інтерпретації прогнозів, а не лише для демонстрації запуску моделі. Аналіз розглядає:

- загальні метрики якості на тестовій вибірці;
- квантилі абсолютної помилки;
- напрям помилки: переоцінка або недооцінка latency;
- помилку за окремими пристроями;
- помилку за dataset;
- залежність помилки від кількості операцій `nor_conv_3x3`;
- приклади найбільших відхилень між `actual_latency` і `predicted_latency`.

Ключові результати:

| Metric | Value |
| --- | ---: |
| Test rows | 53431 |
| MAE | 0.8700 |
| RMSE | 2.1571 |
| MAPE, % | 185.89 |
| Median absolute error | 0.1844 |
| 95% absolute error | 3.6693 |

**Висновок:** модель придатна для первинного ranking конфігурацій, але якість прогнозування суттєво залежить від пристрою. Найбільші помилки спостерігаються для `raspi4` та `pixel3`, тоді як для `eyeriss`, `fpga` та `edgetpu` прогнозування стабільніше. Це означає, що модель добре відтворює загальні закономірності у межах відомих hardware-платформ, але має обмежену переносимість на складніші або повільніші пристрої. Детальний текстовий аналіз наведено у `docs/prediction_result_analysis.md`.

## 10. Обґрунтування вибору моделей

**Файл:** `docs/model_selection_justification.md`

Порівняння трьох моделей має дослідницьку мету:

- `Linear Regression` використовується як проста baseline-модель;
- `HistGradientBoostingRegressor` використовується як сильна модель для табличних даних;
- `MLPRegressor` використовується як нейромережева модель для перевірки складнішої нелінійної апроксимації.

Для `NASBench201 -> latency` отримано такі результати:

| Model | MAE | RMSE | MAPE, % |
| --- | ---: | ---: | ---: |
| Linear Regression | 3.9790 | 6.0433 | 455.17 |
| Gradient Boosting | 0.8746 | 2.1589 | 187.25 |
| MLP Neural Network | 0.8700 | 2.1571 | 185.89 |

Linear Regression має приблизно у `4.55` раза більший MAE, ніж MLP. Це пояснюється тим, що latency не є лінійною функцією від кількості операцій та категоріальних ознак. Вплив архітектури залежить від пристрою, а між ознаками є нелінійні взаємодії, які лінійна модель без додаткових перехресних ознак не описує.

Різниця між Gradient Boosting і MLP мінімальна:

```text
0.8746 - 0.8700 = 0.0046 MAE
```

Тобто MLP є найкращою за числовим значенням MAE, але її перевага над Gradient Boosting практично незначна. Коректний висновок: для цього набору табличних ознак обидві нелінійні моделі працюють майже однаково добре, а Gradient Boosting залишається сильною та стабільною альтернативою. Детальне обґрунтування наведено у `docs/model_selection_justification.md`.

У Streamlit-інтерфейсі рекомендацій використовується не одна універсальна модель, а дві окремі збережені моделі для вибраного search space: одна прогнозує `latency`, друга прогнозує `energy`. Це потрібно тому, що latency та energy є різними цільовими змінними з різними розподілами і різними найкращими моделями. У поточних результатах для latency найкращою є MLP-модель, а для energy - Gradient Boosting. Після отримання двох прогнозів сайт фільтрує конфігурації за обмеженнями `max_latency` і `max_energy` та сортує їх за вибраним критерієм.

Різні найкращі моделі отримано через різну природу цільових змінних і різний обсяг доступних даних. Для latency доступно більше рядків, а для energy значення є лише для частини пристроїв. Тому latency краще апроксимується MLP-моделлю, тоді як для energy стабільнішим за MAE виявився Gradient Boosting. Детальніше це пояснено у `docs/model_selection_justification.md`.
