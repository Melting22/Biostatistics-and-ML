import logging
from pathlib import Path
from typing import final

import numpy as np
from numpy.lib.npyio import NpzFile
from sklearn.impute import KNNImputer, SimpleImputer

from constants import DATA_FILE

log = logging.getLogger(__name__)


@final
class NPZData:
    def __init__(self, npz_file: Path = DATA_FILE) -> None:
        log.info(f"Started loading {npz_file}")
        self.npz: NpzFile = np.load(npz_file, allow_pickle=True)
        log.info(f"Loaded {npz_file}")

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
        log.info(f"Started saving to {output_path}")
        np.savez_compressed(
            output_path,
            data=self.data,
            rows=self.rows,
            cols=self.cols,
        )
        log.info(f"Saved to {output_path}")
        return self

    def fill_all_NaN_with_mean(self) -> NPZData:
        """
        Fills NaN values with mean column values in an NPZ file
        """
        nan_count = np.isnan(self.data).sum()
        print(f"NaN values found: {nan_count}")

        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        self.data = imputer.fit_transform(self.data)
        log.info("Replaced all NaN values with mean column values.")
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

        if len(has_nan) == 0:
            print("No NaN values found")
            return self
        print("--- NaN content table ---")

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

        self.data[:, candidates] = imputed[:, candidates]

        print(f"Imputed columns: {len(candidates)}")
        print(
            f"NaN remaining: {np.isnan(self.data).sum()} (in columns with > 5% NaN)"
        )
        return self

    def drop_columns_with_many_zeros(self, threshold: float = 0.05) -> NPZData:
        """
        Removes columns where the share of zeros exceeds `threshold`.
        """
        zero_counts = (self.data == 0).sum(axis=0)
        keep = zero_counts <= self.patient_count * threshold

        print(f"Columns before: {len(self.cols)}")
        print(f"Columns removed (> {threshold:.0%} zeros): {(~keep).sum()}")

        self.data = self.data[:, keep]
        self.cols = self.cols[keep]
        print(f"Columns after: {len(self.cols)}")
        return self
