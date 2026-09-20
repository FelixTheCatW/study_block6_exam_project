from src.config import (
    CATEGORICAL_COLUMNS,
    CHARTS_DIR,
    CLEAN_DATA_PATH,
    COMBINED_DAILY_PATH,
    CORRELATION_PATH,
    DROP_COLUMNS,
    DIET_DAY_PATH,
    FINAL_REPORT_PATH,
    GROUP_REPORT_PATH,
    LINK_REPORT_PATH,
    MFP_DAY_PATH,
    ML_DATA_DIR,
    PREDICTIONS_PATH,
    PREDICTION_CHART_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.data_analyzer import DataAnalyzer
from src.data_linker import DataLinker
from src.ml_preparer import MLDatasetPreparer
from src.predictor import WeightPredictor
from src.visualizer import Visualizer
from src.report_builder import ReportBuilder


def main() -> None:
    """Run the full DataAnalyzer project."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data.
    print("1. Loading dataset...")
    loader = DataLoader(RAW_DATA_PATH)
    df_raw = loader.load()
    print(f"Raw shape: {df_raw.shape}")

    # 2. Clean data.
    print("2. Cleaning dataset...")
    cleaner = DataCleaner(df_raw)
    df_clean = cleaner.clean(required_columns=REQUIRED_COLUMNS)
    loader.save_dataframe(df_clean, CLEAN_DATA_PATH)
    print(f"Clean shape: {df_clean.shape}")

    # 3. Analyze data.
    print("3. Analyzing dataset...")
    analyzer = DataAnalyzer(df_clean)

    basic_info = analyzer.basic_info()
    weight_stats = analyzer.numeric_statistics(TARGET_COLUMN)
    group_report = analyzer.group_report("activity_type", "calories_burned")
    correlation = analyzer.correlation_with_target(TARGET_COLUMN)

    loader.save_dataframe(group_report.reset_index(), GROUP_REPORT_PATH)
    loader.save_dataframe(correlation.reset_index(), CORRELATION_PATH)

    print("Weight stats:", weight_stats)
    print("Group report (top 5):")
    print(group_report.head())
    print("Correlation with weight (top 5):")
    print(correlation.head())

    # 4. Save charts.
    print("4. Saving charts...")
    visualizer = Visualizer(CHARTS_DIR)

    visualizer.save_histogram(
        dataframe=df_clean,
        column=TARGET_COLUMN,
        filename="weight_histogram.png",
        title="Распределение массы тела (weight_kg)",
    )

    visualizer.save_scatter(
        dataframe=df_clean,
        x_column="daily_steps",
        y_column="calories_burned",
        filename="steps_vs_calories.png",
        title="Шаги и потраченные калории",
    )

    visualizer.save_group_bar(
        group_report=group_report,
        value_column="mean",
        filename="mean_calories_by_activity.png",
        title="Средние потраченные калории по типу тренировки",
    )

    # 5. Prepare dataset for machine learning.
    print("5. Preparing ML dataset...")
    ml_preparer = MLDatasetPreparer(
        dataframe=df_clean,
        target_column=TARGET_COLUMN,
        categorical_columns=CATEGORICAL_COLUMNS,
        drop_columns=DROP_COLUMNS,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = ml_preparer.prepare(
        test_size=TEST_SIZE,
        scale=True,
    )

    ml_preparer.save_prepared_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        output_dir=ML_DATA_DIR,
    )

    ml_shapes = {
        "X_train": X_train.shape,
        "X_test": X_test.shape,
        "y_train": len(y_train),
        "y_test": len(y_test),
    }

    print("ML shapes:", ml_shapes)

    # 6. Train a baseline regression model (demo stage of the diploma).
    print("6. Training baseline regression model...")
    predictor = WeightPredictor(random_state=RANDOM_STATE)
    y_pred, model_metrics = predictor.fit_evaluate(
        X_train, X_test, y_train, y_test
    )
    predictor.save_predictions(y_test, y_pred, PREDICTIONS_PATH)
    visualizer.save_prediction_scatter(
        y_actual=y_test,
        y_pred=y_pred,
        filename="predictions_vs_actual.png",
        title="Прогноз массы тела: факт против прогноза",
    )

    print("Model metrics on test:", model_metrics)
    print("Predictions:", PREDICTIONS_PATH)
    print("Chart:", PREDICTION_CHART_PATH)

    # 7. Link all diploma datasets (MFP + OFF + Health + DietDiary).
    print("7. Linking datasets (MFP + OFF + Health + DietDiary)...")
    mfp_raw = loader.load_mfp()
    off_raw = loader.load_off()
    diet_raw = loader.load_dietdiary()

    linker = DataLinker()
    enriched_mfp = linker.match_mfp_to_off(mfp_raw, off_raw)
    mfp_day = linker.aggregate_mfp_by_day(enriched_mfp)
    health_day = linker.aggregate_health_by_day(df_clean)
    diet_day = linker.aggregate_dietdiary_by_day(diet_raw)
    combined = linker.build_combined_daily(mfp_day, health_day, diet_day)

    loader.save_dataframe(mfp_day, MFP_DAY_PATH)
    loader.save_dataframe(diet_day, DIET_DAY_PATH)
    loader.save_dataframe(combined, COMBINED_DAILY_PATH)

    matched_count = int(enriched_mfp["matched"].sum())
    unique_bases = int(enriched_mfp["base_name"].nunique())
    mean_off_score = float(
        enriched_mfp.loc[enriched_mfp["matched"], "off_score"].mean()
    )
    top_off_categories = enriched_mfp.loc[
        enriched_mfp["matched"], "top_off_category"
    ].value_counts()

    print(
        "Matched meals: "
        f"{matched_count:,} / {len(enriched_mfp):,} "
        f"({matched_count / len(enriched_mfp) * 100:.1f}%)"
    )
    print(f"Combined daily shape: {combined.shape}")

    # 7.1 Chart: OFF categories share.
    visualizer.save_category_bar(
        categories=top_off_categories,
        filename="off_categories_share.png",
        title="Топ категорий Open Food Facts по приёмам пищи",
    )

    # 7.2 Link report.
    link_report_builder = ReportBuilder(LINK_REPORT_PATH)
    link_text = link_report_builder.build_link_report(
        mfp_meals=len(enriched_mfp),
        matched_meals=matched_count,
        base_names=unique_bases,
        mean_score=mean_off_score,
        top_categories=top_off_categories,
        mfp_days=len(mfp_day),
        health_days=len(health_day),
        diet_days=len(diet_day),
        combined_rows=len(combined),
    )
    link_report_builder.save(link_text)
    print("Link report:", LINK_REPORT_PATH)

    # 8. Build report.
    print("8. Building final report...")
    report_builder = ReportBuilder(FINAL_REPORT_PATH)

    insights = [
        "Медианная масса тела по выборке ниже среднего, распределение близко к нормальному.",
        "Средняя калорийность тренировки максимальна у группы " + str(group_report["mean"].idxmax()) + ".",
        "Число шагов и потраченные калории положительно связаны между собой.",
        "Базовая модель (линейная регрессия) объясняет "
        + f"{model_metrics['r2'] * 100:.1f}% дисперсии массы тела "
        + "на тесте с ошибкой RMSE "
        + f"{model_metrics['rmse']:.2f} кг.",
        "Сопоставление блюд MFP со справочником Open Food Facts покрыло "
        + f"{matched_count / len(enriched_mfp) * 100:.1f}% "
        + "приёмов пищи — слои питания и состава продуктов связаны.",
    ]

    report_text = report_builder.build_report(
        title="Блок 6. Финальный проект DataAnalyzer",
        subtitle="Анализ датасета Health and fitness: подготовка данных и базовая модель прогноза массы тела.",
        basic_info=basic_info,
        weight_stats=weight_stats,
        group_report=group_report,
        correlation=correlation,
        ml_shapes=ml_shapes,
        insights=insights,
        model_metrics=model_metrics,
    )
    report_builder.save(report_text)

    print("Project completed.")
    print("Final report:", FINAL_REPORT_PATH)


if __name__ == "__main__":
    main()