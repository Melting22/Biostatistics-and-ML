import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer

from constants import TEMP_DATA_FILE


def fill_NaN_with_mean(
    input_path: Path = TEMP_DATA_FILE,
    output_path: Path = TEMP_DATA_FILE
):
    """
    Fills NaN values with mean column values in an NPZ file
    """
    data = np.load(input_path)
    X = data['data']
    cols = data['cols']
    rows = data['rows']

    print("Data file keys:", data.files)
    print(f"Array shape: {X.shape}")

    nan_count = np.isnan(X).sum()
    print(f"NaN values found: {nan_count}")

    # Imputation: Replacing NaN values with the mean value of its column
    imputer = SimpleImputer(strategy='mean')
    X_clean = imputer.fit_transform(X)

    print("Replaced all NaN values with mean column values.")

    np.savez_compressed(
        output_path,
        data=X_clean,
        rows=rows,
        cols=cols
    )

    print('Saved', output_path)


def main():
    fill_NaN_with_mean()

if __name__ == "__main__":
    main()
