"""Machine learning dataset preparation module.

В блоке 6 модель ещё не обучается: задача — подготовить датасет
так, чтобы в следующем блоке / дипломе его можно было использовать
для регрессии (прогноз weight_kg) или кластеризации (матрица X).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class MLDatasetPreparer:
    """Класс подготовки данных к машинному обучению.

    Что делает класс:
    1. Создаёт новые признаки.
    2. Убирает идентификаторы и производные признаки.
    3. Кодирует категориальные столбцы.
    4. Нормирует числовые признаки.
    5. Делит данные на train/test.
    6. Сохраняет подготовленные CSV-файлы.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        categorical_columns: list[str],
        drop_columns: list[str],
        random_state: int = 42,
    ):
        self.dataframe = dataframe.copy()
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.drop_columns = drop_columns
        self.random_state = random_state

    def add_features(self) -> pd.DataFrame:
        """Создать новые полезные признаки.

        - calories_per_minute — интенсивность траты калорий;
        - activity_load — общая энергетическая нагрузка тренировки (ккал * час);
        - pulse_pressure — пульсовое давление (систолическое минус диастолическое).
        """
        data = self.dataframe.copy()

        data["calories_per_minute"] = (
            data["calories_burned"] / data["duration_minutes"]
        )
        data["activity_load"] = (
            data["calories_burned"] * data["duration_minutes"] / 60
        )
        data["pulse_pressure"] = (
            data["blood_pressure_systolic"] - data["blood_pressure_diastolic"]
        )

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        self.dataframe = data.copy()
        return data

    def make_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Разделить данные на признаки X и целевую переменную y."""
        if self.target_column not in self.dataframe.columns:
            raise ValueError(f"Нет целевой колонки: {self.target_column}")

        data = self.dataframe.drop(columns=[self.target_column])
        for column in self.drop_columns:
            if column in data.columns:
                data = data.drop(columns=[column])

        X = data
        y = self.dataframe[self.target_column]
        return X, y

    def encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Превратить категориальные признаки в числовые 0/1-столбцы."""
        for column in self.categorical_columns:
            if column not in X.columns:
                raise ValueError(f"Нет категориальной колонки: {column}")

        X_encoded = pd.get_dummies(
            X,
            columns=self.categorical_columns,
            drop_first=False,
            dtype=int,
        )
        return X_encoded

    def scale_numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Нормировать числовые признаки.

        Формула: (значение - среднее) / стандартное отклонение.
        Важно для кластеризации и части ML-алгоритмов.
        """
        X_scaled = X.copy()
        numeric_columns = X_scaled.select_dtypes(include="number").columns

        for column in numeric_columns:
            mean_value = X_scaled[column].mean()
            std_value = X_scaled[column].std()
            if std_value == 0:
                X_scaled[column] = 0
            else:
                X_scaled[column] = (X_scaled[column] - mean_value) / std_value

        return X_scaled

    def split_train_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Разделить данные на train и test без scikit-learn."""
        if not 0 < test_size < 1:
            raise ValueError("test_size должен быть между 0 и 1")

        rng = np.random.default_rng(self.random_state)
        indexes = np.arange(len(X))
        rng.shuffle(indexes)

        test_count = int(len(X) * test_size)
        test_indexes = indexes[:test_count]
        train_indexes = indexes[test_count:]

        X_train = X.iloc[train_indexes].reset_index(drop=True)
        X_test = X.iloc[test_indexes].reset_index(drop=True)
        y_train = y.iloc[train_indexes].reset_index(drop=True)
        y_test = y.iloc[test_indexes].reset_index(drop=True)

        return X_train, X_test, y_train, y_test

    def prepare(
        self,
        test_size: float = 0.20,
        scale: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Выполнить полный сценарий подготовки к машинному обучению."""
        self.add_features()

        X, y = self.make_features_and_target()
        X = self.encode_categorical_features(X)

        if scale:
            X = self.scale_numeric_features(X)

        X_train, X_test, y_train, y_test = self.split_train_test(
            X,
            y,
            test_size=test_size,
        )

        if X_train.isna().sum().sum() != 0:
            raise ValueError("В X_train остались пропуски")
        if X_test.isna().sum().sum() != 0:
            raise ValueError("В X_test остались пропуски")

        return X_train, X_test, y_train, y_test

    @staticmethod
    def save_prepared_data(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: Path,
    ) -> None:
        """Сохранить подготовленные данные в CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X_train.to_csv(output_dir / "X_train.csv", index=False)
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_train.to_csv(output_dir / "y_train.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)