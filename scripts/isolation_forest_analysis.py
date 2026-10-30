import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from constants import CHARTS_DIR, REPORTS_DIR
from scripts.npz import NPZData


def isolation_forest(
    npz: NPZData, contamination: float = 0.02, random_state: int = 42
) -> np.ndarray:
    """
    Flags multivariate anomalies with Isolation Forest on standardized data.
    Returns boolean mask of anomalous patients.
    """
    data = npz.data

    # Safety net: fill leftover NaN with column medians
    nan_mask = np.isnan(data)
    if nan_mask.any():
        print(f"Filling {nan_mask.sum()} leftover NaN with column medians")
        medians = np.nanmedian(data, axis=0)
        data = np.where(nan_mask, medians, data)

    print("--- Standardization (StandardScaler) ---")
    scaled = StandardScaler().fit_transform(data)

    print("--- Training Isolation Forest ---")
    iso_forest = IsolationForest(
        contamination=contamination, random_state=random_state
    )
    preds = iso_forest.fit_predict(scaled)
    anomaly_mask = preds == -1
    scores = iso_forest.decision_function(scaled)  # lower = more anomalous

    anomalous_idx = np.where(anomaly_mask)[0]
    anomalous_idx = anomalous_idx[np.argsort(scores[anomalous_idx])]
    anomalous_ids = npz.rows[anomalous_idx].tolist()

    print(f"Anomalous patients found: {len(anomalous_idx)}")
    print("Anomalous patient IDs:", anomalous_ids)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "isolation_forest_anomalies_list.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "status", "score"])
        for i in anomalous_idx:
            writer.writerow([npz.rows[i], "Anomalous", f"{scores[i]:.4f}"])
    print(f"Report saved: {report_path}")

    print("--- Plotting 2D anomaly map (PCA) ---")
    x_2d = PCA(n_components=2).fit_transform(scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        x_2d[~anomaly_mask, 0],
        x_2d[~anomaly_mask, 1],
        c="royalblue",
        alpha=0.85,
        s=70,
        label="Норма / Обычный профиль",
    )
    plt.scatter(
        x_2d[anomaly_mask, 0],
        x_2d[anomaly_mask, 1],
        c="crimson",
        alpha=0.85,
        s=70,
        label="Аномалия (Выброс)",
    )
    plt.title(
        "Многомерные аномалии головного мозга (Isolation Forest + PCA)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Главная компонента 1 (PC1)")
    plt.ylabel("Главная компонента 2 (PC2)")
    plt.legend(title="Статус пациента")

    out_dir = CHARTS_DIR / "IsolationForest"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "visualization_isolation_forest_pca.png", dpi=300)
    plt.close()

    return anomaly_mask
