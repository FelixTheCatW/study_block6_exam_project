"""Dataset linking module.

Связывает слои данных дипломного проекта в единую ежедневную таблицу:

- MFP Diaries (питание): dish_name -> справочник Open Food Facts
  (текстовое сопоставление), дневной intake по БЖУ;
- Health and fitness (расход энергии, активность, сон, вес) по дням;
- DietDiary (вес, стартовый вес, фото/заметки приёмов пищи) по дням.

Между слоями нет общих ключей (разные пользователи и годы), поэтому
объединение строится на уровне гармонизированной дневной схемы:
одинаковые колонки для intake/активности/веса и колонка source.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

# Слова, не несущие смысла для сопоставления (порции, упаковка, общие слова).
STOPWORDS = {
    "a", "an", "and", "the", "in", "of", "on", "with", "without", "fresh",
    "raw", "frozen", "ready", "to", "eat", "from", "my", "brand", "food",
    "drink", "product", "cup", "cups", "oz", "ounce", "ounces", "g", "gram",
    "grams", "ml", "l", "fl", "large", "medium", "small", "x", "2", "3", "1",
    "bag", "bottle", "can", "box", "package", "serving", "servings", "portion",
    "portions", "slice", "slices", "pieces", "piece", "count", "each",
    "homemade", "organic", "day", "instant", "mix", "size",
}

MIN_MATCH_SCORE = 0.5
MAX_CANDIDATES = 300


def normalize_tokens(text: str) -> list[str]:
    """Разбить название на значимые токены (строчные, без стоп-слов)."""
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def parse_dish_base(dish_name: pd.Series) -> pd.Series:
    """Отбросить из названия блюда информацию о порции после запятой."""
    return dish_name.str.split(",").str[0].str.strip()


class DataLinker:
    """Связывает датасеты питания и здоровья в единую дневную таблицу."""

    def __init__(self):
        self.off_keywords = []
        self._off_top_cats = None

    # ------------------------------------------------------------------ #
    # 1) Сопоставление блюд MFP со справочником Open Food Facts            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_off_index(off_df: pd.DataFrame) -> tuple[list[set[str]], dict[str, list[int]]]:
        """Построить инвертированный индекс OFF: токен -> индексы товаров."""
        off_token_sets = []
        index: dict[str, list[int]] = defaultdict(list)

        for i, name in enumerate(off_df["product_name"]):
            token_set = set(normalize_tokens(name))
            off_token_sets.append(token_set)
            for token in token_set:
                index[token].append(i)

        return off_token_sets, index

    @staticmethod
    def _match_base(name: str, off_token_sets, index) -> tuple[int, float]:
        """Найти лучший товар OFF по containment-скору токенов.

        Score = |токены блюда ∩ токены товара| / min(блюдо, товар).
        """
        tokens = set(normalize_tokens(name))
        if not tokens:
            return -1, 0.0

        candidates: set[int] = set()
        for token in tokens:
            candidates.update(index.get(token, ()))
        if not candidates:
            return -1, 0.0

        best_index, best_score = -1, 0.0
        for candidate in candidates:
            candidate_tokens = off_token_sets[candidate]
            if not candidate_tokens:
                continue
            intersection = len(tokens & candidate_tokens)
            score = intersection / min(len(tokens), len(candidate_tokens))
            if score > best_score:
                best_index, best_score = candidate, score

        return best_index, best_score

    def match_mfp_to_off(self, mfp_df: pd.DataFrame, off_df: pd.DataFrame) -> pd.DataFrame:
        """Обогатить приёмы пищи MFP категорией и составом из OFF.

        Возвращает копию mfp_df с колонками:
        base_name, off_index, off_score, off_product, off_top_category, matched.
        """
        off_token_sets, index = self._build_off_index(off_df)
        off_top_cats = (
            off_df["category"].str.split(",").str[0].str.strip()
        )

        enriched = mfp_df.copy()
        enriched["base_name"] = parse_dish_base(enriched["dish_name"])

        unique_bases = enriched["base_name"].unique()
        match_map = {}
        for name in unique_bases:
            match_map[name] = self._match_base(name, off_token_sets, index)

        base_names = enriched["base_name"].map(lambda name: match_map[name][0])
        base_scores = enriched["base_name"].map(lambda name: match_map[name][1])

        enriched["off_index"] = base_names.astype("int64").where(
            base_scores > 0,
            -1,
        )
        enriched["off_score"] = base_scores
        enriched["matched"] = base_scores >= MIN_MATCH_SCORE

        off_product_by_idx = off_df["product_name"].to_numpy()
        off_category_by_idx = off_top_cats.to_numpy()

        enriched["off_product"] = np.where(
            enriched["off_index"] >= 0,
            off_product_by_idx[enriched["off_index"].clip(lower=0)],
            "",
        )
        enriched["top_off_category"] = np.where(
            enriched["off_index"] >= 0,
            off_category_by_idx[enriched["off_index"].clip(lower=0)],
            "Unmatched",
        )

        self._off_top_cats = off_top_cats
        return enriched

    # ------------------------------------------------------------------ #
    # 2) Ежедневная агрегация слоёв                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggregate_mfp_by_day(enriched: pd.DataFrame) -> pd.DataFrame:
        """Агрегировать обогащённые приёмы пищи по (user_id, date)."""
        intake_columns = ["calories", "carbs", "fat", "protein", "sodium", "sugar"]

        daily = (
            enriched.groupby(["user_id", "date"], as_index=False)[intake_columns]
            .sum()
        )
        daily["meals_count"] = enriched.groupby(["user_id", "date"])["dish_name"].count().values
        daily["matched_share"] = (
            enriched.groupby(["user_id", "date"])["matched"].mean().values
        )
        daily["mean_off_score"] = (
            enriched.groupby(["user_id", "date"])["off_score"].mean().values
        )

        # Главная категория OFF по дневной калорийности.
        day_calories_by_category = (
            enriched.groupby(["user_id", "date", "top_off_category"])["calories"]
            .sum()
            .reset_index()
            .sort_values("calories")
            .drop_duplicates(["user_id", "date"], keep="last")
        )
        daily = daily.merge(
            day_calories_by_category[["user_id", "date", "top_off_category"]],
            on=["user_id", "date"],
            how="left",
        )

        return daily

    @staticmethod
    def aggregate_health_by_day(health_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегировать Health and fitness по дате (средние по участникам)."""
        daily = (
            health_df.groupby("date", as_index=False)
            .agg(
                weight_kg=("weight_kg", "mean"),
                bmi=("bmi", "mean"),
                calories_burned=("calories_burned", "mean"),
                daily_steps=("daily_steps", "mean"),
                duration_minutes=("duration_minutes", "mean"),
                sleep_hours=("sleep_hours", "mean"),
                stress_level=("stress_level", "mean"),
                hydration_level=("hydration_level", "mean"),
                age=("age", "mean"),
                participants=("participant_id", "nunique"),
            )
        )
        return daily

    @staticmethod
    def _split_meal_text(meal: str) -> tuple[int, str]:
        """Разделить ячейку на число фото и текстовую заметку.

        Формат: path.jpg;path.jpg|||текст.
        """
        meal = "" if pd.isna(meal) else str(meal)
        photos, _, notes = meal.partition("|||")
        photo_count = 0
        if photos.strip():
            photo_count = len([p for p in photos.split(";") if p.strip()])
        return photo_count, notes.strip()

    def aggregate_dietdiary_by_day(self, diet_df: pd.DataFrame) -> pd.DataFrame:
        """Агрегировать DietDiary по (ID, date): вес, фото, длина заметок."""
        frame = diet_df.copy()
        for meal_column in ("breakfast", "lunch", "supper"):
            parsed = frame[meal_column].map(self._split_meal_text)
            frame[f"{meal_column}_photos"] = parsed.map(lambda pair: pair[0])
            frame[f"{meal_column}_notes"] = parsed.map(lambda pair: pair[1])

        frame["photos_total"] = (
            frame["breakfast_photos"]
            + frame["lunch_photos"]
            + frame["supper_photos"]
        )
        frame["notes_length"] = (
            frame["breakfast_notes"].str.len()
            + frame["lunch_notes"].str.len()
            + frame["supper_notes"].str.len()
        )

        return frame  # остаётся на уровне записей (ID, date) — это уже день

    # ------------------------------------------------------------------ #
    # 3) Единая ежедневная таблица                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_combined_daily(
        mfp_day: pd.DataFrame,
        health_day: pd.DataFrame,
        diet_day: pd.DataFrame,
    ) -> pd.DataFrame:
        """Собрать гармонизированную дневную таблицу из всех слоёв."""
        intake_common = ["calories", "carbs", "fat", "protein", "sodium", "sugar"]

        mfp_rows = mfp_day[["user_id", "date"] + intake_common + [
            "meals_count", "matched_share", "mean_off_score", "top_off_category",
        ]].copy()
        mfp_rows["source"] = "mfp"
        mfp_rows["weight_kg"] = np.nan
        mfp_rows["calories_burned"] = np.nan
        mfp_rows["daily_steps"] = np.nan
        mfp_rows["sleep_hours"] = np.nan
        mfp_rows["stress_level"] = np.nan
        mfp_rows["hydration_level"] = np.nan
        mfp_rows["start_weight"] = np.nan
        mfp_rows["photos_total"] = np.nan
        mfp_rows["notes_length"] = np.nan

        health_rows = health_day[["date"]].copy()
        health_rows["source"] = "health"
        health_rows["calories"] = np.nan
        health_rows["carbs"] = np.nan
        health_rows["fat"] = np.nan
        health_rows["protein"] = np.nan
        health_rows["sodium"] = np.nan
        health_rows["sugar"] = np.nan
        health_rows["meals_count"] = np.nan
        health_rows["matched_share"] = np.nan
        health_rows["mean_off_score"] = np.nan
        health_rows["top_off_category"] = ""
        health_rows["user_id"] = np.nan
        health_rows["weight_kg"] = health_day["weight_kg"]
        health_rows["start_weight"] = np.nan
        health_rows["calories_burned"] = health_day["calories_burned"]
        health_rows["daily_steps"] = health_day["daily_steps"]
        health_rows["sleep_hours"] = health_day["sleep_hours"]
        health_rows["stress_level"] = health_day["stress_level"]
        health_rows["hydration_level"] = health_day["hydration_level"]
        health_rows["photos_total"] = np.nan
        health_rows["notes_length"] = np.nan

        diet_rows = diet_day[["ID", "date", "weight", "start_weight",
                              "photos_total", "notes_length"]].copy()
        diet_rows["source"] = "dietdiary"
        diet_rows["user_id"] = diet_rows["ID"]
        diet_rows["calories"] = np.nan
        diet_rows["carbs"] = np.nan
        diet_rows["fat"] = np.nan
        diet_rows["protein"] = np.nan
        diet_rows["sodium"] = np.nan
        diet_rows["sugar"] = np.nan
        diet_rows["meals_count"] = np.nan
        diet_rows["matched_share"] = np.nan
        diet_rows["mean_off_score"] = np.nan
        diet_rows["top_off_category"] = ""
        diet_rows["weight_kg"] = diet_rows["weight"]
        diet_rows["calories_burned"] = np.nan
        diet_rows["daily_steps"] = np.nan
        diet_rows["sleep_hours"] = np.nan
        diet_rows["stress_level"] = np.nan
        diet_rows["hydration_level"] = np.nan

        columns = [
            "source", "date", "user_id",
            "calories", "carbs", "fat", "protein", "sodium", "sugar",
            "meals_count", "matched_share", "mean_off_score", "top_off_category",
            "calories_burned", "daily_steps",
            "sleep_hours", "stress_level", "hydration_level",
            "weight_kg", "start_weight",
            "photos_total", "notes_length",
        ]

        combined = pd.concat(
            [mfp_rows, health_rows, diet_rows],
            ignore_index=True,
        )[columns]

        return combined