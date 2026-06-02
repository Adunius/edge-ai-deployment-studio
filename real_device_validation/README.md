# Перевірка прогнозування на реальному Android-пристрої

## Мета

Початкова оцінка моделей у проєкті виконувалася на даних HW-NAS-Bench. Це дає відтворюваний benchmark, але не показує, як прогнозування поводиться на фізичному пристрої. Щоб перевірити практичну корисність підходу, було додано окремий експеримент real-device validation.

Мета експерименту:

- запустити вибрані FBNet-блоки на реальному Android-смартфоні;
- виміряти фактичну latency;
- зібрати додаткові resource metrics: CPU time, PSS memory, Java heap і native heap;
- порівняти її з прогнозами моделі;
- оцінити не лише абсолютну похибку, а й те, чи зберігається правильний порядок швидших і повільніших блоків.

## Пристрій

Для перевірки використано фізичний Android-смартфон:

| Параметр | Значення |
| --- | --- |
| Device | LG G8X ThinQ |
| Model | `LM-G850` |
| Android version | 10 |
| Hardware | `mh2lm` |
| Board platform | `msmnile` |

Після розширення експерименту додано другий Android-пристрій:

| Параметр | Значення |
| --- | --- |
| Device | Redmi Note 9 Pro |
| Model | `Redmi Note 9 Pro` |
| Android version | 12 |
| Hardware | `qcom` |
| Board platform | `atoll` |

Підключення виконувалося через Android Debug Bridge. Обидва телефони успішно визначалися командою `adb devices` як авторизовані пристрої.

Важливе обмеження: у HW-NAS-Bench є Android-подібний benchmark-пристрій `pixel3`, але фізично доступним був LG G8X ThinQ. Тому експеримент не трактується як точне відтворення `pixel3` latency. Він використовується як real-device sanity check на реальному Android edge-device.

## Що саме перевірялося

Для real-device validation було обрано FBNet-блоки, тому що вони описуються компактним набором параметрів і їх простіше відтворити у вигляді TFLite-моделей.

Один FBNet-блок описується такими параметрами:

- `input_h`
- `input_w`
- `cin`
- `cout`
- `expansion`
- `kernel`
- `stride`
- `group`

Було вибрано 24 блоки з рядків `device=pixel3`:

- 8 швидких блоків;
- 8 середніх блоків;
- 8 повільних блоків.

Такий вибір потрібен, щоб перевірити не тільки близькі за продуктивністю конфігурації, а весь діапазон latency.

## Реалізований pipeline

Для експерименту додано такі компоненти:

```text
scripts/select_real_device_candidates.py
        |
        v
reports/real_device_candidates_lg_g8x.csv
        |
        v
scripts/generate_fbnet_tflite_models.py
        |
        v
android_benchmark/
        |
        v
reports/real_device_validation_lg_g8x.csv
        |
        v
scripts/analyze_real_device_validation.py
        |
        v
reports/real_device_validation_lg_g8x_summary.json
```

## Етап 1. Вибір кандидатів

Скрипт:

```text
scripts/select_real_device_candidates.py
```

вибирає FBNet-блоки для перевірки та зберігає їх у:

```text
reports/real_device_candidates_lg_g8x.csv
```

Для кожного блоку зберігаються:

- `block_id`;
- benchmark-пристрій `pixel3`;
- реальний пристрій `lg_g8x_thinq`;
- прогнозована latency;
- група latency: `fast`, `medium`, `slow`;
- параметри блоку;
- `block_name`.

## Етап 2. Генерація TFLite-моделей

Скрипт:

```text
scripts/generate_fbnet_tflite_models.py
```

створює TFLite proxy-моделі для вибраних FBNet-блоків. Для цього було використано окреме Python-середовище з TensorFlow:

```text
.venv-tflite
```

Основне середовище проєкту використовує Python 3.14, а TensorFlow було встановлено в окреме середовище на Python 3.12.

Згенеровані моделі записуються в Android-проєкт:

```text
android_benchmark/app/src/main/assets/models/block_*.tflite
android_benchmark/app/src/main/assets/model_index.json
```

Ці моделі є proxy-відтворенням FBNet-блоків на основі параметрів з CSV. Вони не є повною реконструкцією оригінальних benchmark-графів HW-NAS-Bench, тому результат інтерпретується як практична перевірка ranking, а не як точне відтворення benchmark.

## Етап 3. Android benchmark-додаток

Для запуску на телефоні створено Android Studio проєкт:

```text
android_benchmark/
```

Додаток:

- читає `model_index.json`;
- завантажує `.tflite` моделі з assets;
- запускає кожен блок через TensorFlow Lite Interpreter;
- виконує 10 warmup-запусків;
- виконує 50 вимірюваних запусків;
- рахує mean, median, min та max latency;
- вимірює process CPU time під час inference;
- зберігає memory metrics до та після запуску блоку:
  - total PSS;
  - Java heap;
  - native heap;
- зберігає CSV з результатами.

Після запуску на LG G8X ThinQ результат було скопійовано з телефону через `adb pull` у файл:

```text
reports/real_device_validation_lg_g8x.csv
```

## Етап 4. Аналіз результатів

Скрипт:

```text
scripts/analyze_real_device_validation.py
```

аналізує виміряні результати та створює:

```text
reports/real_device_validation_lg_g8x_analyzed.csv
reports/real_device_validation_lg_g8x_summary.json
```

Для кожного блоку додатково обчислюються:

- absolute error;
- relative error;
- predicted rank;
- measured rank;
- rank difference.

Після розширення Android benchmark-додатку CSV також містить resource metrics:

- `cpu_total_ms`;
- `cpu_per_run_ms`;
- `cpu_wall_ratio`;
- `pss_before_kb`;
- `pss_after_kb`;
- `pss_delta_kb`;
- `java_heap_before_kb`;
- `java_heap_after_kb`;
- `java_heap_delta_kb`;
- `native_heap_before_kb`;
- `native_heap_after_kb`;
- `native_heap_delta_kb`.

Ці значення формують невеликий реальний датасет вимірювань на LG G8X ThinQ: для кожного FBNet-блоку є прогнозована latency, фактична latency, CPU usage proxy та memory usage proxy.

## Основні результати

| Metric | Value |
| --- | ---: |
| Devices | 2 |
| Rows | 48 |
| MAE, ms | 3.2112 |
| RMSE, ms | 4.1495 |
| Median absolute error, ms | 2.6813 |
| Median relative error, % | 83.47 |
| Pearson correlation | 0.9036 |
| Spearman rank correlation | 0.9157 |

## Результати за пристроями

| Device | Rows | MAE, ms | RMSE, ms | Pearson | Spearman | Median CPU/run, ms | Median PSS, KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LG G8X ThinQ | 24 | 3.6738 | 4.5597 | 0.9043 | 0.9157 | 6.91 | 55124 |
| Redmi Note 9 Pro | 24 | 2.7486 | 3.6939 | 0.9110 | 0.9322 | 5.57 | 83212.5 |

## Результати за групами

| Group | Rows | Predicted median, ms | Measured median, ms | Median abs. error, ms |
| --- | ---: | ---: | ---: | ---: |
| fast | 16 | 0.0777 | 0.9748 | 0.8405 |
| medium | 16 | 0.9278 | 6.2520 | 5.3741 |
| slow | 16 | 14.1057 | 15.3582 | 2.4676 |

## Інтерпретація Spearman correlation

`Spearman rank correlation` показує, наскільки добре збігається порядок об'єктів. У цьому експерименті вона відповідає на питання:

> Якщо модель прогнозує, що один FBNet-блок швидший за інший, чи справді він швидший на реальному телефоні?

Отримане значення:

```text
Spearman rank correlation = 0.9157
```

є високим. Це означає, що модель добре зберігає порядок швидших і повільніших блоків на реальних Android-пристроях.

Водночас це не означає, що модель з точністю 91.57% вгадує latency у мілісекундах. Абсолютні значення відрізняються через різницю між benchmark-пристроєм `pixel3` і фізичними Android-смартфонами, а також через proxy-відтворення FBNet-блоків у TFLite.

## Висновок

Real-device validation на двох Android-смартфонах показала, що модель не є точним секундоміром для довільного Android-пристрою, але добре працює як інструмент попереднього ranking.

Практичний висновок:

- абсолютні latency на реальних Android-смартфонах відрізняються від прогнозів для `pixel3`;
- швидкі, середні та повільні блоки логічно розділяються;
- порядок блоків добре узгоджується з реальними вимірюваннями;
- створено власний малий real-device dataset: 24 FBNet-блоки на 2 Android-пристроях, тобто 48 реальних вимірювань;
- датасет містить latency, CPU time, PSS memory, Java heap і native heap;
- модель можна використовувати для попереднього відбору FBNet-блоків перед фінальним вимірюванням на конкретному пристрої.

## Власний real-device dataset

Після зауваження щодо ресурсів Android benchmark-додаток було розширено: окрім latency він записує CPU time та memory metrics для кожного блоку. Після запуску на двох телефонах сформовано об'єднаний датасет:

```text
reports/real_android_fbnet_dataset.csv
```

У ньому кожен рядок відповідає запуску одного FBNet-блоку на одному фізичному Android-пристрої. Датасет містить параметри блоку, device metadata, predicted latency, measured latency, CPU usage proxy та memory usage proxy.

Підсумковий файл:

```text
reports/real_android_fbnet_dataset_summary.json
```

містить загальні метрики, метрики окремо за пристроями та summary для CPU/memory usage.
