"""Data loading module.

Модуль отвечает только за загрузку и сохранение данных:
загрузка не смешивается с очисткой, анализом и подготовкой к ML.
"""
from __future__ import annotations

from pathlib import Path

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
        """Загрузить датасет Health and fitness из CSV."""
        if not self.raw_path.exists():
            raise FileNotFoundError(
                f"Датасет не найден: {self.raw_path}. "
                "Положите health_fitness_dataset.csv в папку data/raw."
            )
        return pd.read_csv(self.raw_path)

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