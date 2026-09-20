# Блок 6. Финальный проект DataAnalyzer

## Предметная область

Питание, нутриенты, вес и здоровье. Проект — первая итерация дипломной работы
«Анализ пищевого поведения и прогнозирование показателей питания и веса».

## Цель

Собрать в VS Code рабочий проект анализа данных на Python: загрузить реальный
датасет, очистить его, провести исследовательский анализ, построить графики,
подготовить данные к будущему машинному обучению.

## Датасет

**Health and fitness** (Kaggle, evan65549, лицензия CC0):

https://www.kaggle.com/datasets/evan65549/health-and-fitness-dataset

- 687 701 запись, 23 столбца;
- реальные данные 3 000 участников за 2024 год;
- целевая переменная для регрессии — `weight_kg` (масса тела).

Локальная копия: `data/raw/health_fitness_dataset.csv`.

## Что делает проект

1. Загружает CSV-датасет.
2. Проверяет структуру данных.
3. Очищает пропуски (`health_condition` заполняется категорией `Unknown`,
   числовые пропуски — медианой) и удаляет дубликаты.
4. Считает статистики NumPy/Pandas.
5. Строит групповой отчёт по типу тренировки.
6. Строит графики (гистограмма, scatter, столбчатые диаграммы).
7. Создаёт новые признаки (`calories_per_minute`, `activity_load`, `pulse_pressure`).
8. Кодирует категориальные столбцы в 0/1.
9. Нормирует числовые признаки.
10. Делит данные на train/test.
11. Обучает базовую линейную регрессию (прогноз `weight_kg`) и оценивает на тесте.
12. Сохраняет подготовленные файлы для машинного обучения.
13. Связывает слои дипломного проекта: блюда MFP (6,5 млн приёмов пищи)
    сопоставляются со справочником Open Food Facts (73,9% совпадений),
    а дневные слои MFP + Health + DietDiary объединяются
    в единую ежедневную таблицу `combined_daily.csv`.

## Архитектура

Проект разделён на модули и классы:

| Модуль | Назначение |
|---|---|
| `src/config.py` | пути, имена колонок, параметры проекта |
| `src/data_loader.py` | загрузка и сохранение CSV |
| `src/data_cleaner.py` | очистка пропусков и дубликатов |
| `src/data_analyzer.py` | NumPy/Pandas-статистика, группировки, корреляции |
| `src/data_linker.py` | связка MFP + Open Food Facts + Health + DietDiary |
| `src/visualizer.py` | сохранение графиков |
| `src/ml_preparer.py` | признаки, кодирование, нормирование, train/test split |
| `src/predictor.py` | обучение и оценка базовой модели (scikit-learn) |
| `src/report_builder.py` | формирование итогового markdown-отчёта |
| `main.py` | единый сценарий запуска |

## Как запустить

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
python -m pytest -q
```

После запуска появляются:

- `data/processed/health_clean.csv`
- `data/processed/mfp_day_intake.csv` (дневной intake MFP, обогащённый OFF)
- `data/processed/dietdiary_day.csv` (вес и фото по дням)
- `data/processed/combined_daily.csv` (единая ежедневная таблица 4 слоёв)
- `data/ml/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `data/ml/predictions_test.csv` (факт/прогноз модели на тесте)
- `reports/final_report.md`
- `reports/datasets_link_report.md`
- `reports/group_report_activity_type.csv`
- `reports/correlation_with_weight.csv`
- `reports/charts/*.png` (5 графиков, включая факт против прогноза)

## Связка датасетов

Перечень слоёв дипломного проекта и их роль:

| Слой | Источник | Роль |
|---|---|---|
| Питание | MFP Diaries (`mfp-diaries.parquet`) | дневной intake: калории, БЖУ |
| Справочник | Open Food Facts (`food_data.csv`) | категории и состав на 100 г |
| Здоровье | Health and fitness | расход энергии, активность, вес |
| Дневник с фото | DietDiary (`diet_diary.csv`) | вес по дням, фото приёмов пищи |

DietDiary — официальный датасет репозитория
[yxG1005/Weight_Prediction](https://github.com/yxG1005/Weight_Prediction)
(«Navigating Weight Prediction with Diet Diary», ACM Multimedia 2024 Oral,
arXiv 2408.05445): `data.csv`, `predict_ingr.json`, `DietDiary.zip` (фото).
Формат приёма пищи — `пути к фото;через ;|||ингредиенты через пробел`.

Между слоями нет общих ключей (разные пользователи и годы), поэтому объединение
построено как гармонизированная дневная схема с колонкой `source`.
Блюда MFP сопоставляются со справочником OFF по токенам названий
(`containment >= 0.50`): совпадение для 73,9% приёмов пищи (4,8 млн из 6,5 млн).

## Связь с ИИ

Подготовленная матрица X и целевая переменная `weight_kg` — основа задачи
регрессии на защите диплома. Матрица X без y пригодна для кластеризации.
Единая ежедневная таблица `combined_daily.csv` — вход для объединённого
ML-пайплайна диплома (intake + активность + вес + категории OFF).

### Базовая модель (ветка `feature/ml-prediction`)

Линейная регрессия обучается на `X_train`/`y_train` и оценивается
на `X_test`/`y_test` (прогноз `weight_kg`). Текущие метрики на тесте:
MAE ≈ 4.74 кг, RMSE ≈ 5.56 кг, R2 ≈ 0.578. Результаты в
`data/ml/predictions_test.csv`, график «факт против прогноза» —
`reports/charts/predictions_vs_actual.png`.

## Автор

Конышев Иван Сергеевич