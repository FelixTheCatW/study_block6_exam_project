"""Tests for MLDatasetPreparer."""
import pytest

from src.config import CATEGORICAL_COLUMNS, DROP_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.ml_preparer import MLDatasetPreparer


@pytest.fixture()
def preparer(real_health_sample):
    return MLDatasetPreparer(
        dataframe=real_health_sample,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=42,
    )


def test_ml_preparer_creates_numeric_train_test(preparer):
    X_train, X_test, y_train, y_test = preparer.prepare(
        test_size=TEST_SIZE,
        scale=True,
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert X_train.select_dtypes(exclude="number").shape[1] == 0
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_ml_preparer_drops_derived_columns(preparer):
    preparer.add_features()
    X, y = preparer.make_features_and_target()

    assert TARGET_COLUMN not in X.columns
    for column in DROP_COLUMNS:
        assert column not in X.columns


def test_ml_preparer_adds_features(preparer):
    data = preparer.add_features()

    for feature in ("calories_per_minute", "activity_load", "pulse_pressure"):
        assert feature in data.columns