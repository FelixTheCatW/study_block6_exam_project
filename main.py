"""Main script for Block 6 exam DataAnalyzer project."""
from src.config import (
    CATEGORICAL_COLUMNS,
    CHARTS_DIR,
    CLEAN_DATA_PATH,
    CORRELATION_PATH,
    DROP_COLUMNS,
    FINAL_REPORT_PATH,
    GROUP_REPORT_PATH,
    ML_DATA_DIR,
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
from src.ml_preparer import MLDatasetPreparer
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

    # 6. Build report.
    print("6. Building final report...")
    report_builder = ReportBuilder(FINAL_REPORT_PATH)

    insights = [
        "Медианная масса тела по выборке ниже среднего, распределение близко к нормальному.",
        "Средняя калорийность тренировки максимальна у группы " + str(group_report["mean"].idxmax()) + ".",
        "Число шагов и потраченные калории положительно связаны между собой.",
    ]

    report_text = report_builder.build_report(
        title="Блок 6. Финальный проект DataAnalyzer",
        subtitle="Анализ датасета Health and fitness: подготовка данных к прогнозу массы тела.",
        basic_info=basic_info,
        weight_stats=weight_stats,
        group_report=group_report,
        correlation=correlation,
        ml_shapes=ml_shapes,
        insights=insights,
    )
    report_builder.save(report_text)

    print("Project completed.")
    print("Final report:", FINAL_REPORT_PATH)


if __name__ == "__main__":
    main()