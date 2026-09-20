"""Tests for WeightPredictor."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.config import CATEGORICAL_COLUMNS, DROP_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.ml_preparer import MLDatasetPreparer
from src.predictor import WeightPredictor
from src.visualizer import Visualizer


@pytest.fixture()
def prepared_data(real_health_sample):
    preparer = MLDatasetPreparer(
        dataframe=real_health_sample,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = preparer.prepare(
        test_size=TEST_SIZE,
        scale=True,
    )
    return X_train, X_test, y_train, y_test


def test_predictor_fits_and_predicts(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    predictor = WeightPredictor(random_state=42)

    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)

    assert isinstance(predictor.model, LinearRegression)
    assert len(y_pred) == len(y_test)
    assert y_pred.notna().all()


def test_predictor_metrics_are_plausible(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    predictor = WeightPredictor(random_state=42)

    y_pred, metrics = predictor.fit_evaluate(X_train, X_test, y_train, y_test)

    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert -1.0 <= metrics["r2"] <= 1.0
    assert len(y_pred) == len(y_test)


def test_predictor_saves_predictions(tmp_path):
    y_test = pd.Series([60.0, 65.0, 70.0])
    y_pred = pd.Series([61.0, 64.0, 71.0])
    output_path = tmp_path / "predictions_test.csv"

    WeightPredictor.save_predictions(y_test, y_pred, output_path)

    saved = pd.read_csv(output_path)
    assert list(saved.columns) == ["actual", "prediction"]
    assert saved["actual"].tolist() == [60.0, 65.0, 70.0]


def test_predictor_reads_prepared_data(prepared_data, tmp_path):
    X_train, X_test, y_train, y_test = prepared_data
    ml_dir = tmp_path / "ml"
    ml_dir.mkdir()

    X_train.to_csv(ml_dir / "X_train.csv", index=False)
    X_test.to_csv(ml_dir / "X_test.csv", index=False)
    y_train.to_csv(ml_dir / "y_train.csv", index=False)
    y_test.to_csv(ml_dir / "y_test.csv", index=False)

    loaded = WeightPredictor.read_prepared_data(ml_dir, target_column=TARGET_COLUMN)
    X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = loaded

    assert X_train_loaded.shape == X_train.shape
    assert y_train_loaded.name == TARGET_COLUMN
    assert np.allclose(y_test_loaded, y_test, atol=1e-12)


def test_prediction_scatter_saved(tmp_path, prepared_data):
    X_train, X_test, y_train, y_test = prepared_data
    predictor = WeightPredictor(random_state=42)
    y_pred, _ = predictor.fit_evaluate(X_train, X_test, y_train, y_test)

    visualizer = Visualizer(tmp_path)
    chart_path = visualizer.save_prediction_scatter(
        y_actual=y_test,
        y_pred=y_pred,
        filename="predictions_vs_actual.png",
        title="Факт против прогноза",
    )

    assert chart_path.exists()
    assert chart_path.stat().st_size > 0