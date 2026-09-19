"""Project configuration for Block 6 exam DataAnalyzer.

Датасет: Health and fitness (Kaggle, evan65549 / CC0).
Тема диплома: анализ пищевого поведения и прогноз веса/калорий.
"""
from __future__ import annotations

from pathlib import Path

# Корневая папка проекта.
BASE_DIR = Path(__file__).resolve().parent.parent

# Папки проекта.
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ML_DATA_DIR = DATA_DIR / "ml"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

# Локальный путь к исходному датасету.
RAW_DATA_PATH = RAW_DATA_DIR / "health_fitness_dataset.csv"

# Дополнительные источники дипломного проекта.
MFP_PARQUET_PATH = RAW_DATA_DIR / "mfp-diaries.parquet"
OFF_CSV_PATH = RAW_DATA_DIR / "food_data.csv"
DIETDIARY_CSV_PATH = RAW_DATA_DIR / "diet_diary.csv"

# Первичный источник всех данных диплома (локальная библиотека).
SOURCE_DATA_DIR = Path("D:/study/data")

# Основные выходные файлы.
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "health_clean.csv"
GROUP_REPORT_PATH = REPORTS_DIR / "group_report_activity_type.csv"
CORRELATION_PATH = REPORTS_DIR / "correlation_with_weight.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"

# Файлы связки датасетов.
MFP_DAY_PATH = PROCESSED_DATA_DIR / "mfp_day_intake.csv"
DIET_DAY_PATH = PROCESSED_DATA_DIR / "dietdiary_day.csv"
COMBINED_DAILY_PATH = PROCESSED_DATA_DIR / "combined_daily.csv"
LINK_REPORT_PATH = REPORTS_DIR / "datasets_link_report.md"

# Файлы для будущего машинного обучения.
X_TRAIN_PATH = ML_DATA_DIR / "X_train.csv"
X_TEST_PATH = ML_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = ML_DATA_DIR / "y_train.csv"
Y_TEST_PATH = ML_DATA_DIR / "y_test.csv"

# Колонки датасета Health and fitness.
REQUIRED_COLUMNS = [
    "participant_id",
    "date",
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "bmi",
    "activity_type",
    "duration_minutes",
    "intensity",
    "calories_burned",
    "daily_steps",
    "avg_heart_rate",
    "resting_heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "endurance_level",
    "sleep_hours",
    "stress_level",
    "hydration_level",
    "smoking_status",
    "health_condition",
    "fitness_level",
]

# Целевая переменная для будущей задачи регрессии.
TARGET_COLUMN = "weight_kg"

# Категориальные столбцы для кодирования.
CATEGORICAL_COLUMNS = [
    "gender",
    "activity_type",
    "intensity",
    "smoking_status",
    "health_condition",
]

# Идентификаторы и производные признаки, которые не передаются модели.
DROP_COLUMNS = ["participant_id", "date", "bmi"]

# Параметры повторяемости.
RANDOM_STATE = 42
TEST_SIZE = 0.20