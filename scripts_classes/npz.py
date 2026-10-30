import json
from pathlib import Path
from typing import Any, final

import numpy as np
from numpy.lib.npyio import NpzFile
from sklearn.impute import KNNImputer, SimpleImputer

from constants import DATA_FILE, NOSOLOGIES_FILE


@final
class NPZData:
    def __init__(
        self, npz_file: Path = DATA_FILE, json_file: Path = NOSOLOGIES_FILE
    ) -> None:
        with open(json_file, "r", encoding="utf-8") as f:
            self.json = json.load(f)
        # self.npz: NpzFile = np.load(npz_file, allow_pickle=False)
        self.npz: NpzFile = np.load(npz_file, allow_pickle=True)

        self.data = self.npz["data"]
        self.rows = self.npz["rows"]
        self.cols = self.npz["cols"]

    @property
    def patient_count(self):
        return len(self.rows)

    def save(self, output_path: Path) -> NPZData:
        """
        Writes the data to `output_path`
        """
        # np.savez_compressed(
        #     output_path,
        #     data=np.array(self.data),
        #     rows=np.array(self.rows),
        #     cols=np.array(self.cols),
        # )
        np.savez_compressed(
            output_path,
            data=self.data,
            rows=self.rows,
            cols=self.cols,
        )
        print(f"Saved to {output_path}")
        return self

    def info(self) -> NPZData:
        print(f"Array shape: {self.data.shape}")
        print(f"Patient count: {self.patient_count}")
        return self

    def show_patient_data(self, patient_identifier: int | str = 73) -> NPZData | None:
        """
        Outputs all measurement data on a patient. Accepts index or ID-code.
        """
        if isinstance(patient_identifier, int):
            idx = patient_identifier
            if not 0 <= idx < self.patient_count:
                print(f"Error: index {idx} out of range")
                return None
        else:
            found = np.where(self.rows == patient_identifier)[0]
            if len(found) == 0:
                print(f'Error: patient "{patient_identifier}" not found')
                return None
            idx = found[0]

        for col_idx, col_name in enumerate(self.cols):
            print(f"{col_idx:<4} | {col_name!s:<40} | {self.data[idx, col_idx]}")
        return self

    def check_invalid_negatives(self) -> NPZData:
        allowed = np.zeros(len(self.cols), dtype=bool)
        for token in ("/min", "curv", "кривизн", "asymmetry", "ai"):
            allowed |= np.char.find(self.cols, token) >= 0
        neg_counts = (self.data < 0).sum(axis=0)  # NaN < 0 -> False
        bad = np.where((neg_counts > 0) & ~allowed)[0]

        print(f"Measurements with invalid negative values: {len(bad)}")
        for i in bad[:10]:
            print(f"  - {self.cols[i]}: {neg_counts[i]} invalid negatives")
        return self

    def check_zeros(self) -> NPZData:
        zero_counts = (self.data == 0).sum(axis=0)  # NaN == 0 -> False
        threshold = self.patient_count * 0.05

        idx = np.where(zero_counts > 0)[0]
        less = [i for i in idx if zero_counts[i] <= threshold]
        more = [i for i in idx if zero_counts[i] > threshold]

        print(f"\nTotal patient count: {self.patient_count}")
        print(f"5% of patient count: {threshold:.1f}")

        self._print_zero_table(
            "TABLE 1: Zero values <= 5% (imputation candidates)", less, zero_counts
        )
        self._print_zero_table(
            "TABLE 2: Zero values > 5% (removal candidates)", more, zero_counts
        )
        return self

    def _print_zero_table(self, title: str, indices: Any, zero_counts: Any) -> None:
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
            pct = zero_counts[i] / self.patient_count * 100
            print(f"{i:<8} | {self.cols[i]!s:<70} | {zero_counts[i]:<12} | {pct:.2f}%")
        print(line)

    def check_manually(self) -> NPZData:
        # TODO: Check these patients for zeros and negatives automatically
        sample_size = max(1, int(self.patient_count * 0.05))

        np.random.seed(42)
        random_indices = np.random.choice(
            self.patient_count, size=sample_size, replace=False
        )

        print(
            f"Selected {sample_size} patients for a manual check (5% of {self.patient_count} samples):"
        )
        print("-" * 65)
        print(f"{'№':<4} | {'Код пациента (Rows)':<20} | {'Индекс в массиве'}")
        print("-" * 65)

        for i, idx in enumerate(random_indices, 1):
            patient_id = self.rows[idx]
            print(f"{i:<4} | {patient_id!s:<20} | {idx}")

        print("-" * 65)
        print("""
              Usage: Find these patients in the original data files by their IDs
              Check them manually.
        """)
        return self

    def find_duplicates(self) -> NPZData:
        """
        Find and outputs duplicate rows
        """
        unique_rows, counts = np.unique(self.rows, return_counts=True)
        duplicates = unique_rows[counts > 1]

        if len(duplicates) > 0:
            print(f"Duplicates found: {len(duplicates)}")
            print(duplicates)

            for dup in duplicates:
                count = counts[unique_rows == dup][0]
                print(f'Patient code "{dup}" appears {count} times')
        else:
            print("No duplicates found")
        return self

    def fill_all_NaN_with_mean(self) -> NPZData:
        """
        Fills NaN values with mean column values in an NPZ file
        """
        print("Data file keys:", self.npz.files)
        print(f"Array shape: {self.data.shape}")

        nan_count = np.isnan(self.data).sum()
        print(f"NaN values found: {nan_count}")

        imputer = SimpleImputer(strategy="mean")
        self.data = imputer.fit_transform(self.data)
        print("Replaced all NaN values with mean column values.")
        return self

    def fill_NaN(self, n_neighbors: int = 5) -> NPZData:
        """
        Performs imputation on NaN values in columns where NaN % is less than 5
        """
        nan_counts = np.isnan(self.data).sum(axis=0)
        threshold = self.patient_count * 0.05

        has_nan = np.where(nan_counts > 0)[0]
        candidates = np.array(
            [i for i in has_nan if nan_counts[i] <= threshold], dtype=int
        )

        print("--- NaN content table ---")
        if len(has_nan) == 0:
            print("No NaN values found")
            return self

        line = "-" * 110
        print(f"Columns with NaN: {len(has_nan)}")
        print(line)
        print(
            f"{'Index':<8} | {'Feature':<50} | {'NaN count':<12} | {'%':<10} | Status"
        )
        print(line)
        for i in has_nan:
            pct = nan_counts[i] / self.patient_count * 100
            status = (
                "Imputation candidate (<= 5%)"
                if nan_counts[i] <= threshold
                else "Not an imputation candidate (> 5%)"
            )
            print(
                f"{i:<8} | {self.cols[i]!s:<50} | {nan_counts[i]:<12} | {pct:<9.2f}% | {status}"
            )
        print(line)

        if len(candidates) == 0:
            print("Nothing to impute")
            return self

        print(f"\n--- KNN imputation (n_neighbors={n_neighbors}) ---")
        imputer = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
        imputed = imputer.fit_transform(self.data)

        # Подставляем значения только в столбцы-кандидаты, остальные не трогаем
        self.data[:, candidates] = imputed[:, candidates]

        print(f"Imputed columns: {len(candidates)}")
        print(f"NaN remaining: {np.isnan(self.data).sum()} (in columns with > 5% NaN)")
        return self
