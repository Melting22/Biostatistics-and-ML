from pathlib import Path

import numpy as np

from constants import DATA_FILE


def find_duplicates(data_file: Path = DATA_FILE):
    data = np.load(data_file)
    rows = data["rows"]

    print(f"Data file keys: {data.files}")
    print(f"Patient count: {len(rows)}")

    unique_rows, counts = np.unique(rows, return_counts=True)
    duplicates = unique_rows[counts > 1]

    if len(duplicates) > 0:
        print(f"Duplicates found: {len(duplicates)}")
        print(duplicates)

        for dup in duplicates:
            count = counts[unique_rows == dup][0]
            print(f'Patient code "{dup}" appears {count} times')
    else:
        print("No duplicates found")


def main():
    find_duplicates()


if __name__ == "__main__":
    main()
