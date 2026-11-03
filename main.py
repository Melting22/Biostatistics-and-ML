import logging

from constants import DATA_FILE, NOSOLOGIES_FILE
from scripts.analysis import info
from scripts.iqr_analysis import iqr_analyze
from scripts.isolation_forest_analysis import isolation_forest
from scripts.json_data import JSONData
from scripts.npz import NPZData
from scripts.visualization import create_histogram, create_TSNE

logging.basicConfig(
    filename="logs.txt",
    filemode="w",
    level=logging.DEBUG,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

j = JSONData().from_csv().normalize_nosologies().save(path=NOSOLOGIES_FILE)

n = (
    NPZData()
    .from_jsondata(j)
    .fill_NaN()
    .drop_columns_with_many_zeros()
    .save(path=DATA_FILE)
)
# n = (
#     NPZData()
#     .from_file(NOSOLOGIES_FILE)
#     .fill_NaN()
#     .drop_columns_with_many_zeros()
#     .save()
# )

info(n)

create_TSNE(n)
create_histogram(n)

iqr_analyze(n)
isolation_forest(n)
