import json
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from constants import CHARTS_DIR, NOSOLOGIES_FILE
from scripts.npz import NPZData

TSNE_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]

log = logging.getLogger(__name__)


def create_histogram(npz: NPZData, m_number: int) -> None:
    """Creates a histogram for the specified measurement."""
    if not 0 <= m_number < len(npz.cols):
        print(f"Measurement {m_number} doesn't exist")
        return
    measurement_name = npz.cols[m_number]

    # Plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # KDE histogram
    # stat="density" ensures the histogram and KDE are of the same scale
    # 1. Histogram without KDE
    sns.histplot(
        npz.data[:, m_number],
        stat="density",
        bins=30,
        color="skyblue",
        edgecolor="black",
    )
    # 2. Overlaping with KDE line
    sns.kdeplot(npz.data[:, m_number], color="steelblue", linewidth=2.5)

    # Axis names
    plt.title(f"\n{measurement_name}\n", fontsize=14, fontweight="bold")
    plt.xlabel("Значения", fontsize=12)
    plt.ylabel("Плотность вероятности", fontsize=12)

    save_path = CHARTS_DIR / "histograms" / f"{measurement_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def create_tsne(
    npz: NPZData,
    init: Literal["random", "pca"] = "pca",
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    nosologies_file: Path = NOSOLOGIES_FILE,
) -> None:
    """Creates TSNE charts."""
    data = npz.data  # (501, 4600)

    with nosologies_file.open("r", encoding="utf-8") as f:
        patients_info = json.load(f)

    # perplexity обычно выбирают в пределах от 5 до 50 (для 501 вектора подойдет 30)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        init=init,
    )
    x_2d = tsne.fit_transform(data)

    # Plot regular TSNE
    log.info("Plotting regular TSNE")
    plt.figure(figsize=(10, 8))
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c="blue", alpha=0.7, edgecolors="b")

    plt.title(
        f"t-SNE визуализация векторов ({data.shape[0]} x {data.shape[1]} -> 2D)",
        fontsize=14,
    )
    plt.xlabel("t-SNE Компонента 1", fontsize=12)
    plt.ylabel("t-SNE Компонента 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = (
        CHARTS_DIR
        / "TSNE"
        / f"{init}_{perplexity}_{max_iter}_{random_state}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # Plot colored TSNE
    for criteria in ("gender", "dataset", "nosology"):
        log.info(f"Plotting colored TSNE ({criteria})")
        labels = np.array(
            [patients_info[patient][criteria] for patient in npz.rows]
        )

        unique_labels = np.unique(labels)

        plt.figure(figsize=(10, 8))

        for label, color in zip(unique_labels, TSNE_COLORS, strict=False):
            mask = labels == label

            plt.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                c=[color],
                label=str(label),
                alpha=0.8,
                edgecolors="b",
                s=40,  # Dot size
            )
        plt.title(
            f"t-SNE визуализация векторов ({data.shape[0]} x {data.shape[1]} -> 2D)",
            fontsize=14,
        )
        plt.xlabel("t-SNE Компонента 1", fontsize=12)
        plt.ylabel("t-SNE Компонента 2", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.legend(title="Группы", fontsize=11, title_fontsize=12, loc="best")

        save_path = (
            CHARTS_DIR
            / "TSNE_color"
            / f"{criteria}_{init}_{perplexity}_{max_iter}_{random_state}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
