import logging

from scripts.analysis import info
from scripts.iqr_analysis import iqr_analyze
from scripts.isolation_forest_analysis import isolation_forest
from scripts.npz import NPZData
from scripts.visualization import create_histogram, create_TSNE

logging.basicConfig(
    filename="log.txt",
    filemode="w",
    level=logging.DEBUG,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

n = NPZData().fill_NaN().drop_columns_with_many_zeros()

info(n)

create_TSNE(n)
create_histogram(n)

iqr_analyze(n)
isolation_forest(n)
