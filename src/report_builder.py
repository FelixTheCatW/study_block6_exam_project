"""Report builder module."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportBuilder:
    """Класс для формирования итогового markdown-отчёта."""

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def build_report(
        self,
        title: str,
        subtitle: str,
        basic_info: dict[str, object],
        weight_stats: dict[str, float],
        group_report: pd.DataFrame,
        correlation: pd.Series,
        ml_shapes: dict[str, tuple[int, int] | int],
        insights: list[str],
    ) -> str:
        """Сформировать текст отчёта."""
        top_activity = group_report["mean"].idxmax()
        top_corr_name = correlation.drop(labels=["weight_kg"], errors="ignore").idxmax()
        top_corr_value = correlation.drop(labels=["weight_kg"], errors="ignore").max()

        insights_text = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(insights))

        corr_text = correlation.head(8).to_string()

        report_text = f"""# {title}

{subtitle}

## 1. Данные

Датасет: Health and fitness (Kaggle, evan65549, CC0).

Количество строк после очистки: {basic_info["rows"]}.
Количество колонок после очистки: {basic_info["columns"]}.
Пропусков после очистки: {basic_info["missing_values"]}.

## 2. Статистика целевой переменной (weight_kg)

Минимум: {weight_stats["min"]:.2f}.
Максимум: {weight_stats["max"]:.2f}.
Среднее: {weight_stats["mean"]:.2f}.
Медиана: {weight_stats["median"]:.2f}.
Стандартное отклонение: {weight_stats["std"]:.2f}.

## 3. Групповой анализ

Самая высокая средняя калорийность тренировки у группы: {top_activity}.

Группировка выполнена по колонке `activity_type`.
Целевой показатель: `calories_burned`.

## 4. Корреляция с весом

Сильнее всего с целевой переменной `weight_kg` связан признак:
`{top_corr_name}` ({top_corr_value:.3f}).

Топ корреляций:

{corr_text}

## 5. Аналитические выводы

{insights_text}

## 6. Подготовка к машинному обучению

Созданы новые признаки:

- calories_per_minute;
- activity_load;
- pulse_pressure.

Категориальные столбцы закодированы в 0/1-колонки. BMI исключён из признаков,
потому что он является производной функцией от массы тела.

Данные разделены на train/test:

- X_train: {ml_shapes["X_train"]}
- X_test: {ml_shapes["X_test"]}
- y_train: {ml_shapes["y_train"]}
- y_test: {ml_shapes["y_test"]}

## 7. Как это связано с ИИ

Модель на этом этапе не обучается. Подготовленная матрица признаков X
и целевая переменная y используются для задачи регрессии:
прогнозирование `weight_kg` по образу жизни и активности.

Для кластеризации можно использовать матрицу X без целевой переменной y:
искать группы пользователей по профилю активности и питания.

## 8. Вывод

Проект DataAnalyzer показывает архитектуру реального Python-проекта:
загрузка, очистка, анализ, визуализация, отчёт и подготовка данных
к машинному обучению разделены по классам и модулям.
"""
        return report_text

    def build_link_report(
        self,
        mfp_meals: int,
        matched_meals: int,
        base_names: int,
        mean_score: float,
        top_categories: pd.Series,
        mfp_days: int,
        health_days: int,
        diet_days: int,
        combined_rows: int,
    ) -> str:
        """Сформировать отчёт о связке датасетов."""
        report_text = f"""# Отчёт о связке датасетов

## 1. Задача

Связать четыре слоя данных дипломного проекта в единую ежедневную таблицу:
MFP Diaries (питание), Open Food Facts (справочник продуктов),
Health and fitness (расход энергии и вес), DietDiary (вес и фото приёмов пищи).

## 2. Сопоставление MFP с Open Food Facts

Приёмов пищи в MFP: {mfp_meals:,}.
Уникальных названий блюд: {base_names:,}.
Совпало с товарами справочника (containment >= 0.50): {matched_meals:,}
({matched_meals / mfp_meals * 100:.1f}% приёмов пищи).

Средний скор сопоставления по совпавшим блюдам: {mean_score:.3f}.

## 3. Топ категорий Open Food Facts по приёмам пищи

{top_categories.head(10).map(lambda value: f"{value:,}").rename("приёмов пищи").to_string()}

## 4. Единая ежедневная таблица

- MFP (intake за день): {mfp_days:,} строк;
- Health (активность/вес за день): {health_days:,} строк;
- DietDiary (вес/фото за день): {diet_days:,} строк.

Итого в `combined_daily.csv`: {combined_rows:,} строк.

Схема гармонизирована: колонки intake (calories, carbs, fat, protein, sodium, sugar),
расхода энергии (calories_burned, daily_steps, sleep_hours), веса (weight_kg)
и колонка source (mfp / health / dietdiary).

## 5. Ограничения

Между слоями нет общих ключей: разные пользователи и периоды времени
(MFP 2014–2015, DietDiary 2017–2021, Health 2024). Объединение построено
на уровне гармонизированной дневной схемы. Прямой пользовательский join
станет возможен в дипломной ветке после сбора единого журнала питания.
"""
        return report_text

    def save(self, report_text: str) -> None:
        """Сохранить markdown-отчёт."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report_text, encoding="utf-8")