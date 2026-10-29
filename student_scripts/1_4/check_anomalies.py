from typing import Any

import numpy as np
import pandas as pd
from pathlib import Path

from constants import DATA_FILE


def _check_invalid_negatives(X_df: pd.DataFrame, cols: Any):
    invalid_negatives = []
    for col_idx, col_name in enumerate(cols):
        col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
        col_name_str = str(col_name).lower()

        if (col_data < 0).any():
            is_allowed_negative = (
                "/min" in col_name_str
                or "curv" in col_name_str
                or "кривизн" in col_name_str
                or "asymmetry" in col_name_str
                or "ai" in col_name_str
            )
            if not is_allowed_negative:
                neg_count = (col_data < 0).sum()
                invalid_negatives.append((col_name, neg_count))

    print(f"Measurements with invalid negative values: {len(invalid_negatives)}")
    if len(invalid_negatives) > 0:
        for name, count in invalid_negatives[:10]:
            print(f"  - {name}: {count} invalid negatives")
    else:
        print("No invalid negatives found.")


def _check_zeros(X_df: pd.DataFrame, cols: Any):
    features_with_zeros = []
    for col_idx, col_name in enumerate(cols):
        col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
        zero_count = (col_data == 0).sum()
        if zero_count > 0:
            features_with_zeros.append((col_idx, col_name, zero_count))

    if len(features_with_zeros) > 0:
        print(f"Measurements with zero values: {len(features_with_zeros)}")
        print("-" * 90)
        print(f"{'№':<4} | {'Measurement':<70} | {'Zero count'}")
        print("-" * 90)

        for col_idx, col_name, zero_count in features_with_zeros:
            print(f"{col_idx:<8} | {str(col_name):<70} | {zero_count}")
        print("-" * 90)
    else:
        print("No zeros found")


def _check_manually(rows: Any):
    # TODO: Check these patients for zeros and negatives automatically
    total_patients = len(rows)
    sample_size = max(1, int(total_patients * 0.05))

    np.random.seed(42)
    random_indices = np.random.choice(total_patients, size=sample_size, replace=False)

    print(
        f"Selected {sample_size} patients for a manual check (5% of {total_patients} samples):"
    )
    print("-" * 65)
    print(f"{'№':<4} | {'Код пациента (Rows)':<20} | {'Индекс в массиве'}")
    print("-" * 65)

    for i, idx in enumerate(random_indices, 1):
        patient_id = rows[idx]
        print(f"{i:<4} | {str(patient_id):<20} | {idx}")

    print("-" * 65)
    print("""
          Usage: Find these patients in the original data files by their IDs
          Check them manually.
    """)


def check_anomalies(data_file: Path = DATA_FILE):
    """
    Check for negative and zero values in measurements
    """
    data = np.load(data_file, allow_pickle=True)
    X = data["data"]
    cols = data["cols"]
    rows = data["rows"]
    X_df = pd.DataFrame(X)

    _check_invalid_negatives(X_df, cols)
    _check_zeros(X_df, cols)
    _check_manually(rows)


def main():
    check_anomalies()


if __name__ == "__main__":
    main()
