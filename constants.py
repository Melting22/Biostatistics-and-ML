from pathlib import Path

from numpy import float32

from flash_dir import FLASH_DIR

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

CSV_DIR = FLASH_DIR / "csv_data"
DATA_DIR = FLASH_DIR / "data"

REPORTS_DIR = BASE_DIR / "reports"

CHARTS_DIR = BASE_DIR / "charts"

DATA_FILENAME = "measurements_with_legend.npz"
NOSOLOGIES_FILENAME = "nosologies.json"

DATA_FILE = DATA_DIR / DATA_FILENAME
NOSOLOGIES_FILE = DATA_DIR / NOSOLOGIES_FILENAME
