import csv
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from constants import CHARTS_DIR, REPORTS_DIR
from scripts.npz import NPZData

log = logging.getLogger(__name__)


def isolation_forest(
    npz: NPZData, contamination: float = 0.02, random_state: int = 42
) -> np.ndarray:
    """Performs IsolationForest analysis.

    Flags multivariate anomalies with Isolation Forest on standardized data.
    Returns boolean mask of anomalous patients.
    """
    log.info("Started IsolationForest analysis")
    print("\n--- IsolationForest analysis ---")
    data = npz.data

    # Safety net: fill leftover NaN with column medians
    nan_mask = np.isnan(data)
    if nan_mask.any():
        print(f"Filling {nan_mask.sum()} leftover NaN with column medians")
        medians = np.nanmedian(data, axis=0)
        data = np.where(nan_mask, medians, data)

    scaled = StandardScaler().fit_transform(data)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
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
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "status", "score"])
        for i in anomalous_idx:
            writer.writerow([npz.rows[i], "Anomalous", f"{scores[i]:.4f}"])

    print(f"IsolationForest report saved to {report_path}")

    _plot(scaled, anomaly_mask)
    log.info("Finished IsolationForest analysis")
    return anomaly_mask


def _plot(scaled: NDArray[Any], anomaly_mask: NDArray[Any]) -> None:
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
    out_path = out_dir / "visualization_isolation_forest_pca.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved IsolationForest plot to {out_path}")
