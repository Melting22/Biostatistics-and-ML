## Scripts for processing and exploring TCP/UCLA MRI data

### Core Classes and Data Structures:
- **json_data.py** — parses CSV files containing demographics and MRI metrics into a unified JSON structure, and normalizes nosologies, gender, and age statistics.
- **npz.py** — defines the `NPZData` class for converting JSON data into compact NumPy arrays, saving/loading `.npz` archives, handling missing values imputation (KNN, mean), and filtering sparse columns.

### Analysis:
- **iqr_analysis.py** — performs interquartile range (IQR) outlier detection, generates outlier reports, and builds patient distribution plots.
- **isolation_forest_analysis.py** — detects multivariate anomalies using Isolation Forest on standardized data alongside PCA visualization.
- **analysis.py** — diagnostic utility scripts for inspecting missing values, dataset duplicates, zero-value shares, and invalid negative values.

### Visualization:
- **visualization.py** — generates distribution histograms for specific measurements as well as t-SNE visualizations of the data vectors (with color-coding by gender, dataset, and nosology).
