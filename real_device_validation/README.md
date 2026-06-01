# Перевірка прогнозування на реальному Android-пристрої

## Мета

Початкова оцінка моделей у проєкті виконувалася на даних HW-NAS-Bench. Це дає відтворюваний benchmark, але не показує, як прогнозування поводиться на фізичному пристрої. Щоб перевірити практичну корисність підходу, було додано окремий експеримент real-device validation.

Мета експерименту:

- запустити вибрані FBNet-блоки на реальному Android-смартфоні;
- виміряти фактичну latency;
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

Підключення виконувалося через Android Debug Bridge. Телефон успішно визначався командою `adb devices` як авторизований пристрій.

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

## Основні результати

| Metric | Value |
| --- | ---: |
| Blocks | 24 |
| MAE, ms | 3.6647 |
| RMSE, ms | 4.4853 |
| Median absolute error, ms | 3.1775 |
| Median relative error, % | 84.10 |
| Pearson correlation | 0.9106 |
| Spearman rank correlation | 0.9183 |

## Результати за групами

| Group | Rows | Predicted median, ms | Measured median, ms | Median abs. error, ms |
| --- | ---: | ---: | ---: | ---: |
| fast | 8 | 0.0777 | 1.3473 | 1.2130 |
| medium | 8 | 0.9278 | 5.9448 | 5.0971 |
| slow | 8 | 14.1057 | 16.8496 | 3.4402 |

## Інтерпретація Spearman correlation

`Spearman rank correlation` показує, наскільки добре збігається порядок об'єктів. У цьому експерименті вона відповідає на питання:

> Якщо модель прогнозує, що один FBNet-блок швидший за інший, чи справді він швидший на реальному телефоні?

Отримане значення:

```text
Spearman rank correlation = 0.9183
```

є високим. Це означає, що модель добре зберігає порядок швидших і повільніших блоків на реальному Android-пристрої.

Водночас це не означає, що модель з точністю 91.83% вгадує latency у мілісекундах. Абсолютні значення відрізняються через різницю між benchmark-пристроєм `pixel3` і фізичним LG G8X ThinQ, а також через proxy-відтворення FBNet-блоків у TFLite.

## Висновок

Real-device validation показала, що модель не є точним секундоміром для довільного Android-пристрою, але добре працює як інструмент попереднього ranking.

Практичний висновок:

- абсолютні latency на LG G8X ThinQ відрізняються від прогнозів для `pixel3`;
- швидкі, середні та повільні блоки логічно розділяються;
- порядок блоків добре узгоджується з реальними вимірюваннями;
- модель можна використовувати для попереднього відбору FBNet-блоків перед фінальним вимірюванням на конкретному пристрої.
