import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from constants import CHARTS_DIR, MEASUREMENTS, DATA_FILE


"""Creates a histogram for the specified measurement
"""
def create_histogram(
        m_number: int = 55,
        data_path: Path = DATA_FILE,
    ):
    with np.load(data_path) as archive:
        data = archive['data']
        patients = archive['rows']
        measurements = archive['cols']

    measurement_name = measurements[m_number]
    if measurement_name not in measurements:
        print(f"{measurement_name} measurement doesn't exist")
        return

    # Plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # KDE histogram
    # stat="density" ensures the histogram and KDE are of the same scale
    # 1. Histogram without KDE
    sns.histplot(
        data[:, m_number],
        stat="density",
        bins=30,
        color="skyblue",
        edgecolor="black"
    )
    # 2. Overlaping with KDE line 
    sns.kdeplot(data[:, m_number], color="steelblue", linewidth=2.5)

    # Axis names
    plt.title(f"\n{measurement_name}\n", fontsize=14, fontweight="bold")
    plt.xlabel("Значения", fontsize=12)
    plt.ylabel("Плотность вероятности", fontsize=12)

    plt.show()

    save_path = CHARTS_DIR / "histograms" / f"{measurement_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)


def main():
    create_histogram()

if __name__ == "__main__":
    main()
