"""Data cleaning module."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataCleaner:
    """Класс для проверки и очистки данных.

    В датасете Health and fitness пропуски есть только в health_condition
    (около 71% записей). Числовые пропуски заполняются медианой,
    категориальные — категорией "Unknown".
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def validate_columns(self, required_columns: list[str]) -> None:
        """Проверить, что в таблице есть обязательные колонки."""
        missing_columns = set(required_columns) - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(f"В датасете нет колонок: {sorted(missing_columns)}")

    def fill_missing_numeric(self, column: str) -> None:
        """Заполнить пропуски в числовой колонке медианой."""
        self.dataframe[column] = self.dataframe[column].astype("float64")
        median_value = self.dataframe[column].median()
        self.dataframe[column] = self.dataframe[column].fillna(median_value)

    def fill_missing_categorical(self, column: str) -> None:
        """Заполнить пропуски в категориальной колонке категорией Unknown."""
        self.dataframe[column] = self.dataframe[column].fillna("Unknown")
        self.dataframe[column] = self.dataframe[column].replace("", "Unknown")

    def drop_duplicates(self) -> None:
        """Удалить полностью повторяющиеся строки."""
        self.dataframe = self.dataframe.drop_duplicates()

    def clean(self, required_columns: list[str]) -> pd.DataFrame:
        """Выполнить полный сценарий очистки."""
        self.validate_columns(required_columns)

        numeric_columns = self.dataframe.select_dtypes(include=["number"]).columns
        for column in numeric_columns:
            if self.dataframe[column].isna().any():
                self.fill_missing_numeric(column)

        categorical_columns = self.dataframe.select_dtypes(include=["object", "string"]).columns
        for column in categorical_columns:
            if self.dataframe[column].isna().any() or (self.dataframe[column] == "").any():
                self.fill_missing_categorical(column)

        self.drop_duplicates()

        total_missing = int(self.dataframe.isna().sum().sum())
        if total_missing != 0:
            raise ValueError(f"После очистки остались пропуски: {total_missing}")

        return self.dataframe.copy()