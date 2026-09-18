"""Tests for MLDatasetPreparer."""
from src.config import CATEGORICAL_COLUMNS, DROP_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.data_loader import DataLoader
from src.ml_preparer import MLDatasetPreparer


def test_ml_preparer_creates_numeric_train_test():
    dataframe = DataLoader.create_demo_health_dataset(rows=200, seed=1)

    preparer = MLDatasetPreparer(
        dataframe=dataframe,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=42,
    )

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


def test_ml_preparer_drops_derived_columns():
    dataframe = DataLoader.create_demo_health_dataset(rows=100, seed=7)

    preparer = MLDatasetPreparer(
        dataframe=dataframe,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=42,
    )
    preparer.add_features()
    X, y = preparer.make_features_and_target()

    assert TARGET_COLUMN not in X.columns
    for column in DROP_COLUMNS:
        assert column not in X.columns


def test_ml_preparer_adds_features():
    dataframe = DataLoader.create_demo_health_dataset(rows=100, seed=3)

    preparer = MLDatasetPreparer(
        dataframe=dataframe,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=42,
    )

    data = preparer.add_features()

    for feature in ("calories_per_minute", "activity_load", "pulse_pressure"):
        assert feature in data.columns