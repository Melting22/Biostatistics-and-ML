import csv
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

from constants import CHARTS_DIR, REPORTS_DIR
from scripts.npz import NPZData

log = logging.getLogger(__name__)


def iqr_analyze(npz: NPZData, k: float = 3.0) -> np.ndarray:
    """Performs IQR analysis.

    Counts IQR outliers per patient (k * IQR fences), reports patients
    with outliers in at least 1% of features. Returns counts per patient.
    """
    log.info("Started IQR outlier analysis")
    print("\n--- IQR analysis ---")

    threshold = int(0.01 * npz.data.shape[1])
    q1, q3 = np.nanpercentile(npz.data, [25, 75], axis=0)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    # NaN < x and NaN > x are False, so missing values are never outliers
    outlier_mask = (npz.data < lower) | (npz.data > upper)
    broken_counts = outlier_mask.sum(axis=1)

    suspect_idx = np.where(broken_counts >= threshold)[0]
    suspect_idx = suspect_idx[np.argsort(-broken_counts[suspect_idx], kind="stable")]

    print(f"Total patients: {npz.patient_count}")
    print(f"Patients with >= {threshold} IQR outliers: {len(suspect_idx)}")
    for i in suspect_idx[:20]:
        print(f"  {npz.rows[i]!s:<20} | {broken_counts[i]}")

    report_path = REPORTS_DIR / f"iqr_patients_with_more_than_{threshold}_outliers.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "broken_features_count"])
        for i in suspect_idx:
            writer.writerow([npz.rows[i], broken_counts[i]])
    print(f"IQR report saved to {report_path}")
    _plot(broken_counts, threshold)
    log.info("Finished IQR outlier analysis")

    return broken_counts


def _plot(broken_counts: NDArray[Any], threshold: int) -> None:
    plt.figure(figsize=(12, 6))
    sns.histplot(broken_counts, bins=30, color="teal", kde=True)
    plt.axvline(
        x=threshold,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Порог отсечки ({threshold} выбросов)",
    )
    plt.title(
        "Распределение количества IQR-выбросов на одного пациента",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Количество параметров, вышедших за 'усы'")
    plt.ylabel("Количество пациентов")
    plt.legend()

    out_dir = CHARTS_DIR / "IQR"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "visualization_iqr_patient_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved IQR plot to {out_path}")
