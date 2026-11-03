from pathlib import Path
import sys

from numpy import float32

MEASUREMENTS = [
    "hypothalamic_subunits_volumes.v",
    "lh.aparc.pial",
    "rh.aparc",
    "wmparc",
    "lh.w-g.pct",
    "hipposubfields.rh.T1.v22",
    "rh.BA_exvivo.thresh",
    "rh.aparc.a2009s",
    "vsinus",
    "thalamic-nuclei.lh.v13.T1",
    "lh.aparc",
    "thalamic-nuclei.rh.v13.T1",
    "lh.BA_exvivo",
    "rh.aparc.pial",
    "lh.BA_exvivo.thresh",
    "lh.aparc.a2009s",
    "rh.aparc.DKTatlas",
    "hipposubfields.lh.T1.v22",
    "rh.w-g.pct",
    "amygdalar-nuclei.lh.T1.v22",
    "amygdalar-nuclei.rh.T1.v22",
    "entowm",
    "lh.aparc.DKTatlas",
    "rh.BA_exvivo",
    "aseg",
    "BRAINVOL",
]
VALUES_DTYPE = float32


BASE_DIR = Path(__file__).resolve().parent
_FLASH_DIR_FILE = BASE_DIR / ".flash_dir"

def _validate(raw: str) -> Path | None:
    path = Path(raw.strip()).expanduser()
    if path.is_absolute() and path.is_dir():
        return path.resolve()
    return None


def _prompt_flash_dir() -> Path:
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"{_FLASH_DIR_FILE} not found: run any script interactively "
            + "once or create the file manually"
        )
    while True:
        path = _validate(input("Absolute path to the flash drive: "))
        if path is not None:
            _FLASH_DIR_FILE.write_text(f"{path}\n", encoding="utf-8")
            return path
        print("Not an existing absolute directory, try again.")


def _load_flash_dir() -> Path:
    if not _FLASH_DIR_FILE.is_file():
        return _prompt_flash_dir()
    try:
        text = _FLASH_DIR_FILE.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Failed to read {_FLASH_DIR_FILE}: {exc}") from exc
    lines = [s for s in text.splitlines() if s.strip()]
    path = _validate(lines[0]) if lines else None
    if path is None:
        raise RuntimeError(
            f"Invalid path in {_FLASH_DIR_FILE}: mount the drive "
            + "or delete the file to enter the path again"
        )
    return path

FLASH_DIR = _load_flash_dir()


CSV_DIR = FLASH_DIR / "csv_data"
DATA_DIR = FLASH_DIR / "data"

REPORTS_DIR = BASE_DIR / "reports"

CHARTS_DIR = BASE_DIR / "charts"

DATA_FILENAME = "measurements_with_legend.npz"
NOSOLOGIES_FILENAME = "nosologies.json"

DATA_FILE = DATA_DIR / DATA_FILENAME
NOSOLOGIES_FILE = DATA_DIR / NOSOLOGIES_FILENAME
