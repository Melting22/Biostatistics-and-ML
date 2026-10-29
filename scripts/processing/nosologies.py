import itertools
import json
from pathlib import Path

from constants import TEMP_NOSOLOGIES_FILE


NOSO_MAP = {
    "BIPOLAR": ["BP1", "BP2", "BPI", "BPII", "cyclothymia"],
    "SCHZ": ["SZA", "SZ ", "SZ (ruleout SZA)", "SZ", "SZ or SZA"],
    "ADHD": ["ADHD", "mild ADHD"],
    "CONTROL": ["999"],
}
REVERSE_NOSO_MAP = {value: key for key, arr in NOSO_MAP.items() for value in arr}


def normalize_nosologies(
    input_path: Path = TEMP_NOSOLOGIES_FILE, output_path: Path = TEMP_NOSOLOGIES_FILE
):
    """
    Normalize and summarize patient nosology, gender, and age statistics.
    """
    tcp_nosologies = set()
    ucla_nosologies = set()
    others = set()

    with open(input_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    invalid_count = 0

    noso_counters = {}
    gender_counters = {}
    age_counters = {}
    for key in ["TCP", "UCLA"]:
        noso_counters[key] = dict.fromkeys(NOSO_MAP.keys() | {"OTHER"}, 0)
        gender_counters[key] = dict.fromkeys({"M", "F", "O"}, 0)
        age_counters[key] = dict.fromkeys(
            [str(age) for age in itertools.chain(range(18, 69), [999])], 0
        )

    for patient in data_dict.values():
        if patient["dataset"] == "TCP":
            nosology = patient["tcp_demos"]["Primary_Dx"]
            tcp_nosologies.add(nosology)
            if nosology in REVERSE_NOSO_MAP:
                patient["nosology"] = REVERSE_NOSO_MAP[nosology]
            else:
                patient["nosology"] = "OTHER"
                others.add(nosology)
            patient["gender"] = patient["tcp_demos"]["sex"]
            patient["age"] = patient["tcp_demos"]["Age"]
        elif patient["dataset"] == "UCLA":
            nosology = patient["ucla_participant"]["diagnosis"]
            patient["nosology"] = nosology
            patient["gender"] = patient["ucla_participant"]["gender"]
            patient["age"] = patient["ucla_participant"]["age"]
            ucla_nosologies.add(nosology)
        else:
            raise ValueError("Wow, not a patient")

        patient["valid"] = (
            patient["nosology"] != "OTHER"
            and patient["gender"] != "O"
            and patient["age"] != "999"
        )

        if not patient["valid"]:
            invalid_count += 1
        ds = patient["dataset"]
        nos = patient["nosology"]
        gen = patient["gender"]
        age = patient["age"]
        age_counters[ds][age] += 1
        gender_counters[ds][gen] += 1
        noso_counters[ds][nos] += 1

    print("TCP")
    print(tcp_nosologies)
    print("UCLA")
    print(ucla_nosologies)
    print("NOSOLOGIES")
    print(json.dumps(noso_counters, indent=4, ensure_ascii=False))
    print("GENDERS")
    print(json.dumps(gender_counters, indent=4, ensure_ascii=False))
    print("AGES")
    print(*age_counters.items(), sep="\n")
    print("OTHERS")
    print(others)
    print("ANOMALIES")
    print("TOTAL", len(data_dict), "OTHER", len(others), "INVALID", invalid_count)

    with open(output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(data_dict, json_file, indent=4, ensure_ascii=False)


def main():
    normalize_nosologies()


if __name__ == "__main__":
    main()
