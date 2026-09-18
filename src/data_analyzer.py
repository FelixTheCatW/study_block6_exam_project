"""Data analysis module."""
from __future__ import annotations

import numpy as np
import pandas as pd


class DataAnalyzer:
    """Класс аналитика данных.

    Объединяет операции NumPy и Pandas: числовую статистику,
    describe(), группировки и корреляции с целевой переменной.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe.copy()

    def basic_info(self) -> dict[str, object]:
        """Вернуть базовую информацию о таблице."""
        return {
            "rows": int(self.dataframe.shape[0]),
            "columns": int(self.dataframe.shape[1]),
            "column_names": list(self.dataframe.columns),
            "missing_values": int(self.dataframe.isna().sum().sum()),
        }

    def numeric_statistics(self, column: str) -> dict[str, float]:
        """Посчитать NumPy-статистику по числовой колонке."""
        values = self.dataframe[column].to_numpy(dtype="float64")
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
        }

    def pandas_describe(self) -> pd.DataFrame:
        """Вернуть describe() для числовых колонок."""
        return self.dataframe.describe()

    def group_report(
        self,
        group_column: str,
        value_column: str,
    ) -> pd.DataFrame:
        """Построить групповой отчёт.

        Пример: activity_type -> средние calories_burned.
        """
        report = (
            self.dataframe.groupby(group_column)[value_column]
            .agg(["count", "mean", "min", "max"])
            .sort_values(by="mean", ascending=False)
        )
        return report

    def correlation_with_target(self, target_column: str) -> pd.Series:
        """Посчитать корреляцию числовых признаков с целевой переменной."""
        numeric_df = self.dataframe.select_dtypes(include="number")
        correlation = numeric_df.corr(numeric_only=True)[target_column].sort_values(
            ascending=False
        )
        return correlation