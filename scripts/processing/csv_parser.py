import csv
import json
from pathlib import Path
from typing import Any

from constants import CSV_DIR, MEASUREMENTS, TEMP_NOSOLOGIES_FILE


def _substructure_csv_to_json(
    data: dict[str, Any],
    measurement: str,
    csv_file_path: Path,
    patient_column: str,
    substructure_column: str,
):
    """
    Converts substructure data from `csv_file_path` into JSON and puts it into `data`
    """
    patients = set()
    brain = measurement == "BRAINVOL"
    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=";")
        for row in csv_reader:
            if patient_column not in row or (
                not brain and substructure_column not in row
            ):
                raise ValueError("Important columns not found in CSV file.")
            patient = row[patient_column]
            patients.add(patient)
            if not brain:
                substructure = row[substructure_column]
            else:
                substructure = ""

            attributes = row.copy()
            del attributes[patient_column]
            if not brain:
                del attributes[substructure_column]
            del attributes["source_file"]

            if patient not in data:
                data[patient] = {}
            if measurement not in data[patient]:
                data[patient][measurement] = {}

            if not brain and substructure in data[patient][measurement]:
                raise ValueError(
                    f"Duplicate substructure {substructure} for patient {patient} in {measurement}"
                )
            if brain:
                data[patient][measurement] = attributes
            else:
                if substructure != "unknown":
                    data[patient][measurement][substructure] = attributes

            for key, value in attributes.items():
                value = value.replace(",", ".")
                attributes[key] = value
                try:
                    float(value)
                except ValueError:
                    raise ValueError(
                        f"Wrong format for {value} in {measurement} {substructure}\n{json.dumps(row, indent=4, ensure_ascii=False)}"
                    )
                if value == "nan" and substructure != "unknown":
                    raise ValueError(f"NaN in {measurement} {substructure}")
    print("Parsed", len(patients), "patients from ", measurement)


def _demos_csv_to_json(
    data: dict[str, Any],
    csv_file_path: Path,
    patient_column: str,
    attr_name: str,
    dataset: str,
):
    """
    Converts data from `csv_file_path` into JSON and puts it into `data`
    """
    patients = set()
    with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            if patient_column not in row:
                raise ValueError("Important columns not found in the CSV file.")

            patient = row[patient_column]
            patient = patient.replace("_", "")
            if not patient.startswith("sub-"):
                patient = "sub-" + patient

            attributes = row.copy()
            del attributes[patient_column]
            if patient not in data:
                print(f"Patient {patient} not in data.")
            else:
                data[patient][attr_name] = attributes
                data[patient]["dataset"] = dataset
                patients.add(patient)
    print("Parsed", len(patients), "patients from", attr_name)


def _parse_dataset(data: dict[str, Any], dataset_name: str):
    """
    Converts substructure measurements from `dataset_name` into JSON and puts it into `data`
    """
    for m in MEASUREMENTS:
        path = CSV_DIR / f"{dataset_name}_CSV" / f"{dataset_name}_{m}.csv"
        _substructure_csv_to_json(data, m, path, "sub_name", "StructName")
    print("Total", len(data), "patients.")


def convert_csv_to_json(output_path: Path = TEMP_NOSOLOGIES_FILE):
    """
    Converts all CSV data into a single JSON and writes it to `output_path`
    """
    data = {}

    _parse_dataset(data, "TCP")
    _parse_dataset(data, "UCLA")

    _demos_csv_to_json(
        data, CSV_DIR / "TCP_demos_fixed.csv", "subjectkey", "tcp_demos", "TCP"
    )
    _demos_csv_to_json(
        data,
        CSV_DIR / "UCLA_demographics_fixed.csv",
        "participant_id",
        "ucla_demos",
        "UCLA",
    )
    _demos_csv_to_json(
        data,
        CSV_DIR / "UCLA_participants_fixed.csv",
        "participant_id",
        "ucla_participant",
        "UCLA",
    )

    with open(output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def main():
    convert_csv_to_json()


if __name__ == "__main__":
    main()
