from pathlib import Path

import numpy as np
import pandas as pd

from constants import DATA_FILE


def show_patient_data(patient_identifier: int | str = 73, data_file: Path = DATA_FILE):
    """
    Outputs all measurement data on a patient. Accepts index or ID-code.
    """
    data = np.load(data_file, allow_pickle=True)
    X = data["data"]
    cols = data["cols"]
    rows = data["rows"]
    X_df = pd.DataFrame(X)

    if isinstance(patient_identifier, int):
        idx = patient_identifier
        if idx < 0 or idx >= len(rows):
            print(f"Error: index {idx} out of range [0; {len(rows) - 1}]")
            return
    else:
        matching_indices = np.where(rows == patient_identifier)[0]
        if len(matching_indices) == 0:
            print(f'Error: patient "{patient_identifier}" not found')
            return
        idx = matching_indices[0]

    patient_id = rows[idx]
    print(f"\nPatient [{idx}] = {patient_id}:")
    print("-" * 65)
    print(f"{'№':<4} | {'Measurement (column)':<40} | {'Value'}")
    print("-" * 65)

    for col_idx, col_name in enumerate(cols):
        val = X_df.iloc[idx, col_idx]
        print(f"{col_idx:<4} | {col_name!s:<40} | {val}")

    print("-" * 65)


def main():
    show_patient_data()


if __name__ == "__main__":
    main()
