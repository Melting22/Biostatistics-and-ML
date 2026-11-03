from typing import Any

import numpy as np

from scripts.npz import NPZData


def show_patient_data(npz: NPZData, patient_identifier: int | str = 73):
    """
    Outputs all measurement data on a patient. Accepts index or ID-code.
    """
    if isinstance(patient_identifier, int):
        idx = patient_identifier
        if not 0 <= idx < npz.patient_count:
            print(f"Error: index {idx} out of range")
            return
    else:
        found = np.where(npz.rows == patient_identifier)[0]
        if len(found) == 0:
            print(f'Error: patient "{patient_identifier}" not found')
            return
        idx = found[0]

    for col_idx, col_name in enumerate(npz.cols):
        print(f"{col_idx:<4} | {col_name!s:<40} | {npz.data[idx, col_idx]}")


def find_duplicates(npz: NPZData) -> bool:
    """
    Finds duplicate rows
    """
    unique_rows, counts = np.unique(npz.rows, return_counts=True)
    duplicates = unique_rows[counts > 1]

    if len(duplicates) > 0:
        print(f"Duplicates found: {len(duplicates)}")
        print(duplicates)

        for dup in duplicates:
            count = counts[unique_rows == dup][0]
            print(f'Patient code "{dup}" appears {count} times')
    else:
        print("No duplicates found")
    return len(duplicates) > 0


def check_invalid_negatives(npz: NPZData):
    """
    Finds positive-only measurements where negative values occur
    """
    # TODO: Find out what measurements can and can not be negative
    # TODO: Revise the logic
    names = np.char.lower(npz.cols.astype(str))
    allowed = np.zeros(len(names), dtype=bool)
    for token in ("/min", "curv", "кривизн", "asymmetry", "ai"):
        allowed |= np.char.find(names, token) >= 0
    neg_counts = (npz.data < 0).sum(axis=0)  # NaN < 0 -> False
    bad = np.where((neg_counts > 0) & ~allowed)[0]

    print(f"Measurements with invalid negative values: {len(bad)}")
    for i in bad[:10]:
        print(f"  - {npz.cols[i]}: {neg_counts[i]} invalid negatives")
    return bad


def check_zeros(npz: NPZData):
    """
    Prints info on zero value shares in columns
    """
    zero_counts = (npz.data == 0).sum(axis=0)  # NaN == 0 -> False
    threshold = npz.patient_count * 0.05

    idx = np.where(zero_counts > 0)[0]
    less = [i for i in idx if zero_counts[i] <= threshold]
    more = [i for i in idx if zero_counts[i] > threshold]

    print(f"\nTotal patient count: {npz.patient_count}")
    print(f"5% of patient count: {threshold:.1f}")

    _print_zero_table(
        npz,
        "TABLE 1: Zero values <= 5% (imputation candidates)",
        less,
        zero_counts,
    )
    _print_zero_table(
        npz, "TABLE 2: Zero values > 5% (removal candidates)", more, zero_counts
    )


def _print_zero_table(npz: NPZData, title: str, indices: Any, zero_counts: Any):
    print(f"\n--- {title} ---")
    if len(indices) == 0:
        print("No measurements found")
        return
    print(f"Measurements found: {len(indices)}")
    line = "-" * 100
    print(line)
    print(f"{'Index':<8} | {'Measurement':<70} | {'Zero count':<12} | Zero %")
    print(line)
    for i in indices:
        pct = zero_counts[i] / npz.patient_count * 100
        print(
            f"{i:<8} | {npz.cols[i]!s:<70} | {zero_counts[i]:<12} | {pct:.2f}%"
        )
    print(line)


def check_manually(npz: NPZData):
    # TODO: Check these patients for zeros and negatives automatically
    sample_size = max(1, int(npz.patient_count * 0.05))

    np.random.seed(42)
    random_indices = np.random.choice(
        npz.patient_count, size=sample_size, replace=False
    )

    print(
        f"Selected {sample_size} patients for a manual check (5% of {npz.patient_count} samples):"
    )
    print("-" * 65)
    print(f"{'№':<4} | {'Код пациента (Rows)':<20} | {'Индекс в массиве'}")
    print("-" * 65)

    for i, idx in enumerate(random_indices, 1):
        patient_id = npz.rows[idx]
        print(f"{i:<4} | {patient_id!s:<20} | {idx}")

    print("-" * 65)
    print("""
          Usage: Find these patients in the original data files by their IDs
          Check them manually.
    """)
