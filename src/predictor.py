"""Prediction module: baseline regression model.

Ветка feature/ml-prediction: демо-этап диплома. Подготовленный датасет
(X_train/X_test, y_train/y_test) используется для обучения линейной
регрессии — прогноз weight_kg по образу жизни и активности.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class WeightPredictor:
    """Класс обучения и оценки модели линейной регрессии."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = LinearRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "WeightPredictor":
        """Обучить модель на train-выборке."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Спрогнозировать целевую переменную."""
        return pd.Series(self.model.predict(X), index=X.index, name="prediction")

    @staticmethod
    def evaluate(y_actual: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        """Посчитать метрики качества: MAE, RMSE, R2."""
        return {
            "mae": float(mean_absolute_error(y_actual, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
            "r2": float(r2_score(y_actual, y_pred)),
        }

    @staticmethod
    def read_prepared_data(
        ml_dir: Path,
        target_column: str = "weight_kg",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Прочитать подготовленные данные из data/ml."""
        ml_dir = Path(ml_dir)

        X_train = pd.read_csv(ml_dir / "X_train.csv")
        X_test = pd.read_csv(ml_dir / "X_test.csv")

        y_train = pd.read_csv(ml_dir / "y_train.csv")
        y_test = pd.read_csv(ml_dir / "y_test.csv")

        if target_column not in y_train.columns:
            raise ValueError(
                f"Нет целевой колонки {target_column!r} в y_train.csv. "
                f"Доступны: {list(y_train.columns)}"
            )

        y_train = y_train[target_column]
        y_test = y_test[target_column]

        return X_train, X_test, y_train, y_test

    def fit_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Обучить модель и оценить на тесте: (прогнозы, метрики)."""
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)
        return y_pred, metrics

    @staticmethod
    def save_predictions(
        y_test: pd.Series,
        y_pred: pd.Series,
        output_path: Path,
        actual_column: str = "actual",
        predicted_column: str = "prediction",
    ) -> None:
        """Сохранить таблицу факт/прогноз."""
        table = pd.DataFrame(
            {
                actual_column: y_test.reset_index(drop=True),
                predicted_column: y_pred.reset_index(drop=True),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)