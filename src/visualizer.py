"""Visualization module."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


class Visualizer:
    """Класс для сохранения графиков."""

    def __init__(self, charts_dir: Path):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def save_histogram(
        self,
        dataframe: pd.DataFrame,
        column: str,
        filename: str,
        title: str,
    ) -> Path:
        """Сохранить гистограмму."""
        output_path = self.charts_dir / filename

        plt.figure(figsize=(8, 5))
        plt.hist(dataframe[column], bins=40, color="#4C72B0")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("Количество")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()

        return output_path

    def save_scatter(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        filename: str,
        title: str,
    ) -> Path:
        """Сохранить scatter plot."""
        output_path = self.charts_dir / filename

        plt.figure(figsize=(8, 5))
        plt.scatter(
            dataframe[x_column],
            dataframe[y_column],
            alpha=0.15,
            s=8,
            color="#55A868",
        )
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()

        return output_path

    def save_group_bar(
        self,
        group_report: pd.DataFrame,
        value_column: str,
        filename: str,
        title: str,
    ) -> Path:
        """Сохранить столбчатую диаграмму по групповому отчёту."""
        output_path = self.charts_dir / filename

        plt.figure(figsize=(9, 5))
        group_report[value_column].plot(kind="bar", color="#C44E52")
        plt.title(title)
        plt.xlabel(group_report.index.name)
        plt.ylabel(value_column)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()

        return output_path