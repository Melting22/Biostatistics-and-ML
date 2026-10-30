import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from constants import CHARTS_DIR, DATA_FILE, NOSOLOGIES_FILE
from scripts_classes.npz import NPZData

TSNE_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]


def create_histogram(npz: NPZData, m_number: int = 55):
    """
    Creates a histogram for the specified measurement
    """
    measurement_name = npz.cols[m_number]
    if measurement_name not in npz.cols:
        print(f"{measurement_name} measurement doesn't exist")
        return

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

    plt.show()

    save_path = CHARTS_DIR / "histograms" / f"{measurement_name}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)


def create_TSNE(
    npz: NPZData,
    init: str = "pca",
    perplexity: int = 30,
    max_iter: int = 1000,
    random_state: int = 42,
    data_file: Path = DATA_FILE,
    nosologies_file: Path = NOSOLOGIES_FILE,
):
    """
    Creates TSNE charts
    """
    data = np.load(data_file)
    X = data["data"]  # (501, 4600)

    with open(nosologies_file, "r", encoding="utf-8") as f:
        npz.json = json.load(f)

    # perplexity обычно выбирают в пределах от 5 до 50 (для 501 вектора хорошо подойдет 30)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=max_iter,
        init=init,
    )
    X_2d = tsne.fit_transform(X)

    # Plot regular TSNE
    print("Plotting regular TSNE")
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, edgecolors="b", cmap="viridis")

    plt.title(
        f"t-SNE визуализация векторов ({X.shape[0]} x {X.shape[1]} -> 2D)",
        fontsize=14,
    )
    plt.xlabel("t-SNE Компонента 1", fontsize=12)
    plt.ylabel("t-SNE Компонента 2", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.show()

    save_path = (
        CHARTS_DIR / "TSNE" / f"{init}_{perplexity}_{max_iter}_{random_state}.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)

    # Plot colored TSNE
    for criteria in ("gender", "dataset", "nosology"):
        print(f"Plotting colored TSNE ({criteria})")
        labels = np.array([npz.json[patient][criteria] for patient in npz.rows])

        unique_labels = np.unique(labels)

        plt.figure(figsize=(10, 8))

        for label, color in zip(unique_labels, TSNE_COLORS):
            mask = labels == label

            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[color],
                label=str(label),
                alpha=0.8,
                edgecolors="b",
                s=40,  # Dot size
            )
        # plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7, edgecolors='b', cmap='viridis')

        plt.title(
            f"t-SNE визуализация векторов ({X.shape[0]} x {X.shape[1]} -> 2D)",
            fontsize=14,
        )
        plt.xlabel("t-SNE Компонента 1", fontsize=12)
        plt.ylabel("t-SNE Компонента 2", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.legend(title="Группы", fontsize=11, title_fontsize=12, loc="best")

        # plt.show()
        save_path = (
            CHARTS_DIR
            / "TSNE_color"
            / f"{criteria}_{init}_{perplexity}_{max_iter}_{random_state}.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
