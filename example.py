import logging

from constants import DATA_FILE, NOSOLOGIES_FILE
from scripts.iqr_analysis import iqr_analyze
from scripts.isolation_forest_analysis import isolation_forest
from scripts.json_data import JSONData
from scripts.npz import NPZData
from scripts.visualization import create_histogram, create_tsne

logging.basicConfig(
    filename="logs.txt",
    filemode="w",
    level=logging.DEBUG,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

j = (
    JSONData()
    # Load data from files in csv_data directory
    .from_csv()
    # Normalize data
    .normalize_nosologies()
    # Save to a json file
    .save(path=NOSOLOGIES_FILE)
)

n = (
    NPZData()
    # Convert nosologies json data into NPZ
    .from_jsondata(j)
    # Deal with NaN values
    .fill_nan()
    # Deal with columns with too many zeros
    .drop_columns_with_many_zeros()
    # Save to an NPZ file
    .save(path=DATA_FILE)
)

# Save as above, loading from a saved NPZ file
n = (
    NPZData()
    .from_file(DATA_FILE)
    .fill_nan()
    .drop_columns_with_many_zeros()
    .save()
)

j.info()
n.info()

create_tsne(n)
create_histogram(n, 0)

iqr_analyze(n)
isolation_forest(n)
