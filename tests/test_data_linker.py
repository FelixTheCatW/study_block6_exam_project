"""Tests for DataLinker."""
import numpy as np
import pandas as pd

from src.data_linker import DataLinker


def _make_off() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_name": ["Coffee", "Banana", "Chicken Breast"],
            "category": [
                "Beverages, Hot drinks, Coffees",
                "Fruits, Tropical fruits, Bananas",
                "Meats and their products, Poultry, Chickens",
            ],
            "energy_kcal_100g": [1.0, 89.0, 165.0],
            "fat_100g": [0.0, 0.3, 3.6],
            "sugars_100g": [0.0, 12.2, 0.0],
            "carbohydrates_100g": [0.0, 22.8, 0.0],
            "serving_size_g": [100.0, 100.0, 100.0],
        }
    )


def _make_mfp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "date": ["2014-09-14", "2014-09-14", "2014-09-15"],
            "meal_sequence": [1, 1, 1],
            "dish_name": [
                "Coffee - Brewed from grounds, 2 cup (8 fl oz)",
                "Chicken Breast - Grilled, 100 g",
                "Python Snake Soup, 1 bowl",
            ],
            "calories": [5.0, 165.0, 300.0],
            "carbs": [0.0, 0.0, 25.0],
            "fat": [0.0, 3.6, 15.0],
            "protein": [0.0, 31.0, 20.0],
            "sodium": [0.0, 70.0, 1000.0],
            "sugar": [0.0, 0.0, 5.0],
        }
    )


def test_matcher_matches_known_products():
    linker = DataLinker()
    enriched = linker.match_mfp_to_off(_make_mfp(), _make_off())

    assert bool(enriched.loc[0, "matched"]) is True
    assert enriched.loc[0, "top_off_category"] == "Beverages"
    assert bool(enriched.loc[1, "matched"]) is True
    assert enriched.loc[1, "top_off_category"] == "Meats and their products"
    assert bool(enriched.loc[2, "matched"]) is False
    assert enriched.loc[2, "top_off_category"] == "Unmatched"


def test_aggregate_mfp_by_day_sums_intake():
    linker = DataLinker()
    enriched = linker.match_mfp_to_off(_make_mfp(), _make_off())
    daily = linker.aggregate_mfp_by_day(enriched)

    assert len(daily) == 2
    row = daily[daily["user_id"] == 1].iloc[0]
    assert row["calories"] == 170.0
    assert row["protein"] == 31.0
    assert row["meals_count"] == 2
    assert row["matched_share"] == 1.0
    assert row["top_off_category"] in {"Beverages", "Meats and their products"}


def test_aggregate_dietdiary_parses_photos_and_notes():
    linker = DataLinker()
    diet = pd.DataFrame(
        {
            "ID": ["u1", "u1"],
            "date": ["2020/11/17", "2020/11/18"],
            "breakfast": [
                "breakfast/a_1.jpg;breakfast/a_2.jpg|||egg toast",
                "breakfast/b_1.jpg|||",
            ],
            "lunch": ["lunch/a_1.jpg|||soup", ""],
            "supper": ["", "supper/b_1.jpg|||fish"],
            "start_weight": [78.8, 78.4],
            "weight": [78.4, 79.3],
        }
    )

    parsed = linker.aggregate_dietdiary_by_day(diet)

    assert parsed.loc[0, "photos_total"] == 3
    assert parsed.loc[0, "notes_length"] == len("egg toast") + len("soup")
    assert parsed.loc[1, "photos_total"] == 2


def test_build_combined_daily_structure():
    linker = DataLinker()

    mfp_day = pd.DataFrame(
        {
            "user_id": [1],
            "date": ["2014-09-14"],
            "calories": [170.0],
            "carbs": [0.0],
            "fat": [3.6],
            "protein": [31.0],
            "sodium": [70.0],
            "sugar": [0.0],
            "meals_count": [2],
            "matched_share": [1.0],
            "mean_off_score": [0.9],
            "top_off_category": ["Beverages"],
        }
    )
    health_day = pd.DataFrame(
        {
            "date": ["2024/1/1"],
            "weight_kg": [61.2],
            "calories_burned": [220.0],
            "daily_steps": [7000.0],
            "sleep_hours": [7.2],
            "stress_level": [4.0],
            "hydration_level": [2.0],
        }
    )
    diet_day = pd.DataFrame(
        {
            "ID": ["u1"],
            "date": ["2020/11/17"],
            "weight": [78.4],
            "start_weight": [78.8],
            "photos_total": [3],
            "notes_length": [10],
        }
    )

    combined = linker.build_combined_daily(mfp_day, health_day, diet_day)

    assert set(combined["source"].unique()) == {"mfp", "health", "dietdiary"}
    assert combined.shape[0] == 3
    for column in (
        "calories", "calories_burned", "weight_kg", "daily_steps",
        "sleep_hours", "photos_total", "top_off_category",
    ):
        assert column in combined.columns
    mfp_row = combined[combined["source"] == "mfp"].iloc[0]
    assert mfp_row["calories"] == 170.0
    assert np.isnan(mfp_row["weight_kg"])