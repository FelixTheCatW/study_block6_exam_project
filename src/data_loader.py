"""Data loading module.

Модуль отвечает только за загрузку и сохранение данных:
загрузка не смешивается с очисткой, анализом и подготовкой к ML.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DIETDIARY_CSV_PATH,
    MFP_PARQUET_PATH,
    OFF_CSV_PATH,
    SOURCE_DATA_DIR,
)

PATH_SOURCE_DIR = SOURCE_DATA_DIR


class DataLoader:
    """Класс для загрузки CSV-датасета Health and fitness."""

    def __init__(self, raw_path: Path):
        self.raw_path = Path(raw_path)

    def load(self) -> pd.DataFrame:
        """Загрузить датасет.

        Алгоритм:
        1. Если CSV уже есть в data/raw, читаем его.
        2. Если файла нет, создаём учебный fallback-датасет с той же схемой колонок.
        """
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)

        if self.raw_path.exists():
            return pd.read_csv(self.raw_path)

        data = self.create_demo_health_dataset()
        data.to_csv(self.raw_path, index=False)
        return data

    @staticmethod
    def _resolve_source(local_path: Path, source_file_name: str) -> Path:
        """Вернуть локальный путь, либо путь в общей библиотеке данных."""
        if Path(local_path).exists():
            return Path(local_path)
        return PATH_SOURCE_DIR / source_file_name

    def load_mfp(self) -> pd.DataFrame:
        """Загрузить MyFitnessPal Diaries (Parquet)."""
        path = self._resolve_source(MFP_PARQUET_PATH, "mfp-diaries.parquet")
        return pd.read_parquet(path)

    def load_off(self) -> pd.DataFrame:
        """Загрузить справочник продуктов Open Food Facts."""
        path = self._resolve_source(OFF_CSV_PATH, "food_data.csv")
        return pd.read_csv(path, encoding="utf-8")

    def load_dietdiary(self) -> pd.DataFrame:
        """Загрузить дневник питания DietDiary с фото и весом."""
        path = self._resolve_source(DIETDIARY_CSV_PATH, "diet_diary.csv")
        return pd.read_csv(path, encoding="utf-8")

    @staticmethod
    def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
        """Сохранить DataFrame в CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    @staticmethod
    def create_demo_health_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
        """Создать учебный датасет, похожий по схеме на Health and fitness.

        Это запасной вариант для запуска без локального CSV (например, в CI).
        Для экзамена используется реальный датасет.
        """
        rng = np.random.default_rng(seed)

        genders = np.random.choice(
            ["F", "M", "Other"], size=rows, p=[0.49, 0.49, 0.02]
        )
        height_cm = np.where(genders == "M", rng.normal(176, 7, rows), rng.normal(164, 7, rows))
        height_cm = np.clip(height_cm, 140, 210)

        bmi_values = rng.normal(24.5, 4.5, rows)
        bmi_values = np.clip(bmi_values, 14, 42)
        weight_kg = bmi_values * ((height_cm / 100) ** 2)

        activity_types = [
            "Walking",
            "Running",
            "Cycling",
            "Swimming",
            "Yoga",
            "HIIT",
            "Weight Training",
        ]

        data = pd.DataFrame(
            {
                "participant_id": np.arange(1, rows + 1),
                "date": pd.date_range("2024-01-01", periods=rows, freq="h").strftime("%Y/%m/%d %H:%M"),
                "age": rng.integers(18, 66, size=rows),
                "gender": genders,
                "height_cm": height_cm.round(1),
                "weight_kg": weight_kg.round(2),
                "bmi": bmi_values.round(2),
                "activity_type": rng.choice(activity_types, size=rows),
                "duration_minutes": rng.uniform(15, 120, rows).round(1),
                "intensity": rng.choice(["Low", "Medium", "High"], size=rows),
                "calories_burned": rng.uniform(50, 900, rows).round(1),
                "daily_steps": rng.integers(1000, 25000, size=rows),
                "avg_heart_rate": rng.integers(55, 180, size=rows),
                "resting_heart_rate": rng.uniform(45, 95, rows).round(1),
                "blood_pressure_systolic": rng.uniform(90, 180, rows).round(1),
                "blood_pressure_diastolic": rng.uniform(55, 120, rows).round(1),
                "endurance_level": rng.uniform(3, 16, rows).round(2),
                "sleep_hours": rng.uniform(3, 12, rows).round(1),
                "stress_level": rng.integers(1, 11, size=rows),
                "hydration_level": rng.uniform(0.5, 3.5, rows).round(1),
                "smoking_status": rng.choice(["Never", "Former", "Current"], size=rows),
                "health_condition": rng.choice(
                    ["", "Hypertension", "Diabetes", "Asthma"],
                    size=rows,
                    p=[0.7, 0.15, 0.1, 0.05],
                ),
                "fitness_level": rng.uniform(0.01, 25, rows).round(2),
            }
        )
        return data