import csv
import itertools
import json
import logging
from pathlib import Path
from typing import Any

from constants import (
    CSV_DIR,
    MEASUREMENTS,
    NOSOLOGIES_FILE,
)

NOSO_MAP = {
    "BIPOLAR": ["BP1", "BP2", "BPI", "BPII", "cyclothymia"],
    "SCHZ": ["SZA", "SZ ", "SZ (ruleout SZA)", "SZ", "SZ or SZA"],
    "ADHD": ["ADHD", "mild ADHD"],
    "CONTROL": ["999"],
}
REVERSE_NOSO_MAP = {value: key for key, arr in NOSO_MAP.items() for value in arr}

log = logging.getLogger(__name__)


class JSONData:
    """Nosologies data in JSON format."""

    def __init__(self) -> None:
        """Initializes `JSONData` object with empty values."""
        self.data: dict[str, Any] = {}

        self.tcp_nosologies: set[str] = set()
        self.ucla_nosologies: set[str] = set()
        self.others: set[str] = set()
        self.noso_counters: dict[str, dict[str, int]] = {}
        self.gender_counters: dict[str, dict[str, int]] = {}
        self.age_counters: dict[str, dict[str, int]] = {}
        self.invalid_count: int = 0

    def info(self) -> JSONData:
        """Prints various info about the data."""
        # TODO: Export this information to `REPORTS_DIR`
        print("\n--- JSONData info ---")
        self._print_info_list("TCP", sorted(self.tcp_nosologies))
        self._print_info_list("UCLA", sorted(self.ucla_nosologies))
        self._print_info_list("OTHER", sorted(self.others))

        print("NOSOLOGIES")
        self._print_info_dict("NOSOLOGIES", self.noso_counters)
        print("GENDERS")
        self._print_info_dict("GENDERS", self.gender_counters)
        print("AGES")
        for k, table in self.age_counters.items():
            print(f"{k}: ")
            for age, cnt in sorted(table.items(), key=lambda x: -x[1]):
                print(f" - {age}: {cnt}")
                if cnt == 0:
                    break
        print("ANOMALIES")
        print(f"TOTAL {len(self.data)}")
        print(f"OTHER {len(self.others)}")
        print(f"INVALID {self.invalid_count}")
        return self

    def _print_info_list(self, title: str, li: list[Any]) -> None:
        print(f"\n{title}")
        for i, elem in enumerate(li):
            print(f"{i + 1}. {elem}")

    def _print_info_dict(self, title: str, di: dict[Any, Any]) -> None:
        print(f"\n{title}")
        print(json.dumps(di, indent=4, ensure_ascii=False))

    def save(self, path: Path = NOSOLOGIES_FILE) -> JSONData:
        """Saves nosologies data to a JSON file."""
        log.info(f"Started saving to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(self.data, json_file, indent=4, ensure_ascii=False)
        log.info(f"Saved to {path}")
        return self

    def from_csv(self) -> JSONData:
        """Extracts data from files in `CSV_DIR`."""
        log.info("Started parsing CSV files")
        self._parse_dataset("TCP")
        self._parse_dataset("UCLA")

        self._demos_csv_to_json(
            CSV_DIR / "TCP_demos_fixed.csv",
            "subjectkey",
            "tcp_demos",
            "TCP",
        )
        self._demos_csv_to_json(
            CSV_DIR / "UCLA_demographics_fixed.csv",
            "participant_id",
            "ucla_demos",
            "UCLA",
        )
        self._demos_csv_to_json(
            CSV_DIR / "UCLA_participants_fixed.csv",
            "participant_id",
            "ucla_participant",
            "UCLA",
        )
        log.info("Finished parsing CSV files")
        return self

    def normalize_nosologies(self) -> JSONData:
        """Normalizes patient nosology, gender, and age statistics."""
        self.tcp_nosologies = set()
        self.ucla_nosologies = set()
        self.others = set()
        self.noso_counters = {}
        self.gender_counters = {}
        self.age_counters = {}
        self.invalid_count = 0

        for key in ["TCP", "UCLA"]:
            self.noso_counters[key] = dict.fromkeys(NOSO_MAP.keys() | {"OTHER"}, 0)
            self.gender_counters[key] = dict.fromkeys({"M", "F", "O"}, 0)
            self.age_counters[key] = dict.fromkeys(
                [str(age) for age in itertools.chain(range(18, 69), [999])], 0
            )

        for patient in self.data.values():
            if patient["dataset"] == "TCP":
                nosology = patient["tcp_demos"]["Primary_Dx"]
                self.tcp_nosologies.add(nosology)
                if nosology in REVERSE_NOSO_MAP:
                    patient["nosology"] = REVERSE_NOSO_MAP[nosology]
                else:
                    patient["nosology"] = "OTHER"
                    self.others.add(nosology)
                patient["gender"] = patient["tcp_demos"]["sex"]
                patient["age"] = patient["tcp_demos"]["Age"]
            elif patient["dataset"] == "UCLA":
                nosology = patient["ucla_participant"]["diagnosis"]
                patient["nosology"] = nosology
                patient["gender"] = patient["ucla_participant"]["gender"]
                patient["age"] = patient["ucla_participant"]["age"]
                self.ucla_nosologies.add(nosology)
            else:
                raise ValueError("Wow, not a patient")

            patient["valid"] = (
                patient["nosology"] != "OTHER"
                and patient["gender"] != "O"
                and patient["age"] != "999"
            )

            if not patient["valid"]:
                self.invalid_count += 1
            ds = patient["dataset"]
            nos = patient["nosology"]
            gen = patient["gender"]
            age = patient["age"]
            self.age_counters[ds][age] += 1
            self.gender_counters[ds][gen] += 1
            self.noso_counters[ds][nos] += 1
        log.info("Finished normalizing values")
        return self

    def _substructure_csv_to_json(
        self,
        measurement: str,
        csv_file_path: Path,
        patient_column: str,
        substructure_column: str,
    ) -> None:
        patients = set()
        brain = measurement == "BRAINVOL"
        with csv_file_path.open(encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=";")
            for row in csv_reader:
                if patient_column not in row or (
                    not brain and substructure_column not in row
                ):
                    raise ValueError("Important columns not found in CSV file.")
                patient = row[patient_column]
                patients.add(patient)
                substructure = row[substructure_column] if not brain else ""

                attributes = row.copy()
                del attributes[patient_column]
                if not brain:
                    del attributes[substructure_column]
                del attributes["source_file"]

                if patient not in self.data:
                    self.data[patient] = {}
                if measurement not in self.data[patient]:
                    self.data[patient][measurement] = {}

                if not brain and substructure in self.data[patient][measurement]:
                    raise ValueError(
                        f"Duplicate substructure {substructure} "
                        + f"for patient {patient} in {measurement}"
                    )
                if brain:
                    self.data[patient][measurement] = attributes
                else:
                    if substructure != "unknown":
                        self.data[patient][measurement][substructure] = attributes

                for key, value in attributes.items():
                    value = value.replace(",", ".")
                    attributes[key] = value
                    try:
                        float(value)
                    except ValueError:
                        raise ValueError(  # noqa: B904
                            f"Wrong format for {value} "
                            + f"in {measurement} {substructure}\n"
                            + f"{json.dumps(row, indent=4, ensure_ascii=False)}"
                        )
                    if value == "nan" and substructure != "unknown":
                        raise ValueError(f"NaN in {measurement} {substructure}")
        log.info(
            f"Parsed {len(patients)} patients from {measurement} from {csv_file_path}"
        )

    def _demos_csv_to_json(
        self,
        csv_file_path: Path,
        patient_column: str,
        attr_name: str,
        dataset: str,
    ) -> None:
        patients = set()
        not_found = set()
        with csv_file_path.open(encoding="utf-8") as csv_file:
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
                if patient not in self.data:
                    not_found.add(patient)
                else:
                    self.data[patient][attr_name] = attributes
                    self.data[patient]["dataset"] = dataset
                    patients.add(patient)
        print(f"Patients not found in data: {sorted(not_found)}")
        log.info(f"Parsed {len(patients)} patients from {attr_name} from {csv_file_path}")

    def _parse_dataset(self, dataset_name: str) -> None:
        for m in MEASUREMENTS:
            path = CSV_DIR / f"{dataset_name}_CSV" / f"{dataset_name}_{m}.csv"
            self._substructure_csv_to_json(m, path, "sub_name", "StructName")
        print("Parsed total of", len(self.data), "patients.")
        log.info(f"Parsed total of: {len(self.data)} patients")
