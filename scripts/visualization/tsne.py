import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

from constants import CHARTS_DIR, DATA_FILE, NOSOLOGIES_FILE


# Цветовая палитра (можно использовать встроенные, например, 'tab10' или 'Set1')
# COLORS = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
# COLORS = ['#FF2400', '#0047AB', '#00A550']
COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]


def create_TSNE(
    init="pca",
    perplexity=30,
    max_iter=1000,
    random_state=42,
    data_file: Path = DATA_FILE,
    nosologies_file: Path = NOSOLOGIES_FILE,
):
    """
    Creates TSNE charts from the data
    """
    data = np.load(data_file)
    X = data["data"]  # (501, 4600)

    print(f"Data file keys: {data.files}")
    print(f"Array shape: {X.shape}")

    with open(nosologies_file, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

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
        f"t-SNE визуализация векторов ({X.shape[0]} x {X.shape[1]} -> 2D)", fontsize=14
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
    for criteria in {"gender", "dataset", "nosology"}:
        print(f"Plotting colored TSNE ({criteria})")
        labels = np.array([data_dict[patient][criteria] for patient in data["rows"]])

        unique_labels = np.unique(labels)

        plt.figure(figsize=(10, 8))

        for label, color in zip(unique_labels, COLORS):
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


def main():
    create_TSNE()


if __name__ == "__main__":
    main()
