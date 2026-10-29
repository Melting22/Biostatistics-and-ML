import json
import numpy as np
from pathlib import Path

from constants import MEASUREMENTS, TEMP_DATA_FILE, TEMP_NOSOLOGIES_FILE


def convert_json_to_npz(
    input_path: Path = TEMP_NOSOLOGIES_FILE, output_path: Path = TEMP_DATA_FILE
):
    """
    Converts JSON data to Numpy array and saves it as NPZ
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    patients = {}
    column_names = set()
    for patient_key, patient in data_dict.items():
        patients[patient_key] = {}
        for measurement in MEASUREMENTS:
            # TODO: Repair dataset and remove this
            if measurement not in patient.keys():
                print(f"{measurement} is missing!")
                continue
            # TODO: End of remove

            if measurement != "BRAINVOL":
                for subarea, subarea_data in patient[measurement].items():
                    if measurement + subarea == "rh.aparc.DKTatlastemporalpole":
                        continue
                    for key, value in subarea_data.items():
                        if key not in {"NumVert", "SegId", "NVoxels"}:
                            column_name = f"{measurement}/{subarea}/{key}"
                            column_names.add(column_name)
                            patients[patient_key][column_name] = value
            else:
                for key, value in patient[measurement].items():
                    if key not in {"BrainSegVol", "MaskVol"}:
                        column_name = f"BRAINVOL/{key}"
                        column_names.add(column_name)
                        patients[patient_key][column_name] = value

    for patient_key, patient_dict in patients.items():
        for column_name in column_names:
            if column_name not in patient_dict:
                print(column_name, "is missing for", patient_key)
                patient_dict[column_name] = "nan"

    patients_legend = list(patients.keys())
    columns_legend = sorted(column_names)

    num_patients = len(patients_legend)
    num_columns = len(columns_legend)

    data = np.zeros((num_patients, num_columns), dtype=np.float32)
    for i, patient_name in enumerate(patients_legend):
        for j, column_name in enumerate(columns_legend):
            data[i][j] = float(patients[patient_name][column_name])
    np.savez_compressed(
        output_path, data=data, rows=patients_legend, cols=columns_legend
    )

    print("Saved", num_patients, "patients", num_columns, "columns")


def main():
    convert_json_to_npz()


if __name__ == "__main__":
    main()
