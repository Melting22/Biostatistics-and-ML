## Scripts for processing and exploring TCP/UCLA MRI data

### Processing:
 - csv_parser.py - converts CSVs into a single JSON file
 - nosologies.py - normalizes nosologies in JSON
 - tables.py - converts data to Numpy array and saves as NPZ
 - fill_gaps.py - fills the NaN values with mean column values
### Visualization:
 - histogram.py - draws histogram for the specified measurement
 - tsne.py - draws t-SNE charts of the data

## TODO:
- Remove code sections from `csv_parser.py` and `tables.py`
- Run all scripts on uncorrupted `UCLA_lh.aparc.A2009s.csv` and `UCLA_rh.BA_exvivo.csv`
