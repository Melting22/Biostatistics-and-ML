from pathlib import Path
from config import PLOTS_DIR, SRC_NPZ

import pandas as pd
import numpy as np
import seaborn as sns
import diptest
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from scipy.stats import median_abs_deviation, skew, kurtosis

#Returns DataFrame with KNN imputed percentile
#p_left, p_right - left and right percentile
def iqr(df,p_left = 5, p_right = 95):
    result = df.copy()
    numeric = result.select_dtypes(include = 'number')

    for col in numeric.columns:
        values = numeric[col]

        low = np.nanpercentile(values, p_left)
        high = np.nanpercentile(values, p_right)

        outliers = (values < low) | (values > high)
        numeric.loc[outliers, col] = np.nan

    imputer =  KNNImputer(n_neighbors=5, weights='distance')
    imputed = imputer.fit_transform(numeric)

    result.loc[:,numeric.columns] = imputed
    return result

def impute_zeros_or_drop_columns(df, max_zero_share=0.05, n_neighbors=5):
    if not 0 <= max_zero_share <= 1:
        raise ValueError("max_zero_share have to be > 0 and < 1")
    if n_neighbors < 1:
        raise ValueError("n_neighbors have to be >=1")

    result = df.copy()
    numeric_cols = result.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        return result

    numeric = result.loc[:, numeric_cols].copy()

    # percent of zeros
    zero_share = numeric.eq(0).mean()

    # dropping numeric columns, if there are too much zeros
    cols_to_drop = zero_share[zero_share > max_zero_share].index
    result = result.drop(columns=cols_to_drop)

    cols_to_impute = numeric_cols.difference(cols_to_drop)
    numeric = result.loc[:, cols_to_impute].copy()
    numeric = numeric.replace(0, np.nan)

    if len(cols_to_impute) > 0:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        result.loc[:, cols_to_impute] = imputer.fit_transform(numeric)

    return result

#Returns indexed top percent of cols, which Modified Z-score is more than border
def modified_z_anomaly_cols(df, border, percent):
    numeric = df.select_dtypes(include = 'number')
    scores = pd.DataFrame(index = df.index, columns = numeric.columns, dtype = float)
    for col in numeric.columns:
        x = numeric[col]
        median = x.median()
        mad = median_abs_deviation(x.dropna())

        #In this case Modified Z-score could not be applied
        if pd.isna(mad) or mad == 0:
            continue

        scores[col] = 0.6745 * (x - median) / mad

    anomalies = scores.abs() > border
    anomalies_percentage = anomalies.sum() / len(anomalies) * 100
    return anomalies_percentage[anomalies_percentage > percent].index

#Returns Sarle's bimodality coefficient.
#If BC >= 0.55 feature considered as bimodal, otherwise as unimodal
def bimodality_coefficient(data):
    n = len(data)
    g = skew(data)
    k = kurtosis(data, fisher=True)

    return (g**2 + 1) / (k + 3 * ((n-1)**2) / ((n-2) * (n-3)))

#Analyzing bimodality according to Sarle's BC and Harigan's diptest
#Returns DataFrame with results
def analyze_bimodality(df, show_plots=False):
    numeric = df.select_dtypes(include='number')
    results = []

    for col in numeric.columns:
        bc = bimodality_coefficient(numeric[col])
        dip, p_value = diptest.diptest(numeric[col])

        results.append({
            'column': col,
            'BC': bc,
            'BC_bimodal': True if bc >= 0.555 else False,
            'DIP': dip,
            'p_value': p_value,
            'DIP_bimodal': True if p_value < 0.05 else False
        })

    result_df = pd.DataFrame(results)

    # Some plots, if needed
    save_dir = Path("plots")
    if show_plots:

        sorted_df = result_df.dropna(subset=['BC']).sort_values('BC', ascending=False)
        top100 = sorted_df.head(100)  #Top 100 by BC

        for _, row in top100.iterrows():
            col = row['column']
            bc_val = row['BC']

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data=numeric[col].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title(f'{col} — BC: {bc_val:.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Count')
            fig.tight_layout()

            base_filename = col.replace('/', '_')

            fig.savefig(PLOTS_DIR / f'{base_filename}.png', dpi=100, bbox_inches='tight')
            plt.close(fig)  #Free memory
    return result_df

tensor = np.load(SRC_NPZ)
data = pd.DataFrame(tensor['data'], index = tensor['rows'], columns = tensor['cols'])
before_zeros = data.copy()
data = impute_zeros_or_drop_columns(data, 0.05, 5)
data = iqr(data, 5, 95) #IQR and imputation
data = data.drop(columns=modified_z_anomaly_cols(data, 5, 5)) #Modified Z-score and drop

#Are all values unique?
#are_all_unique = 1 if len(set(tensor['rows'])) == len(tensor['rows']) else 0
print(data)


bmdata = analyze_bimodality(data, True)
print(bmdata.iloc[:,5].sum()) #checking number of bimodal features


