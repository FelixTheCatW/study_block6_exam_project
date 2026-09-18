"""Tests for DataCleaner."""
import pandas as pd
import pytest

from src.config import REQUIRED_COLUMNS
from src.data_cleaner import DataCleaner


def _make_dataframe(rows: int = 3) -> pd.DataFrame:
    template = {
        "participant_id": 1,
        "date": "2024/1/1",
        "age": 30,
        "gender": "F",
        "height_cm": 165.0,
        "weight_kg": 60.0,
        "bmi": 22.0,
        "activity_type": "Running",
        "duration_minutes": 30.0,
        "intensity": "Medium",
        "calories_burned": 200.0,
        "daily_steps": 8000,
        "avg_heart_rate": 120,
        "resting_heart_rate": 65.0,
        "blood_pressure_systolic": 120.0,
        "blood_pressure_diastolic": 80.0,
        "endurance_level": 8.0,
        "sleep_hours": 7.0,
        "stress_level": 4,
        "hydration_level": 2.0,
        "smoking_status": "Never",
        "health_condition": "",
        "fitness_level": 5.0,
    }
    frame = pd.DataFrame([template.copy() for _ in range(rows)])
    frame.loc[1, "health_condition"] = None
    frame.loc[1, "weight_kg"] = None
    return frame


def test_cleaner_fills_missing_values():
    dataframe = _make_dataframe()
    assert dataframe["health_condition"].isna().any()
    assert dataframe["weight_kg"].isna().any()

    cleaned = DataCleaner(dataframe).clean(REQUIRED_COLUMNS)

    assert cleaned.isna().sum().sum() == 0
    assert cleaned["health_condition"].isin(["Unknown"]).all()


def test_cleaner_drops_duplicates():
    dataframe = _make_dataframe(rows=3)
    dataframe = pd.concat([dataframe, dataframe.iloc[:2]], ignore_index=True)
    assert len(dataframe) == 5

    cleaned = DataCleaner(dataframe).clean(REQUIRED_COLUMNS)

    assert len(cleaned) == len(cleaned.drop_duplicates())
    assert len(cleaned) < len(dataframe)


def test_cleaner_validates_columns():
    dataframe = _make_dataframe().drop(columns=["weight_kg"])

    with pytest.raises(ValueError):
        DataCleaner(dataframe).clean(REQUIRED_COLUMNS)