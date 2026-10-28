from csv_parser import convert_csv_to_json
from nosologies import normalize_nosologies
from tables import convert_json_to_npz
from fill_gaps import fill_NaN_with_mean

from constants import DATA_DIR, TEMP_DATA_FILE, TEMP_NOSOLOGIES_FILE


"""Processes CSV data into an NPZ dataset with imputation, copies the results into `MASTER_DATA_DIR`
"""
def main():
    convert_csv_to_json()
    normalize_nosologies()
    convert_json_to_npz()
    fill_NaN_with_mean()

    TEMP_DATA_FILE.copy_into(DATA_DIR)
    TEMP_NOSOLOGIES_FILE.copy_into(DATA_DIR)

if __name__ == "__main__":
    main()
