from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from constants import DATA_FILE


def fill_NaN(input_file: Path = DATA_FILE, output_file: Path = DATA_FILE) -> Path:
    data = np.load(input_file, allow_pickle=True)
    X = data["data"]
    cols = data["cols"]
    rows = data["rows"]
    X_df = pd.DataFrame(X, columns=cols, index=rows)
    total_patients = len(rows)

    print(f"Array shape: {X_df.shape}")
    print(f"Patient count: {total_patients}\n")

    # 2. Анализ и вывод таблицы процентного соотношения NaN по столбцам
    nan_stats = []
    threshold_5_pct = total_patients * 0.05

    for col_idx, col_name in enumerate(cols):
        col_data = pd.to_numeric(X_df[col_name], errors="coerce")
        nan_count = col_data.isna().sum()
        nan_pct = (nan_count / total_patients) * 100

        if nan_count > 0:
            status = (
                "Imputation candidate (<= 5%)"
                if nan_count <= threshold_5_pct
                else "Not an Imputation candidate 5%"
            )
            nan_stats.append((col_idx, col_name, nan_count, nan_pct, status))

    print("--- NaN content table ---")
    if len(nan_stats) > 0:
        print(f"Найдено столбцов с пропусками (NaN): {len(nan_stats)}")
        print("-" * 110)
        print(
            f"{'Индекс':<8} | {'Название признака':<50} | {'Кол-во NaN':<12} | {'Процент (%)':<12} | {'Статус'}"
        )
        print("-" * 110)
        for idx, name, count, pct, status in nan_stats:
            print(f"{idx:<8} | {name!s:<50} | {count:<12} | {pct:<11.2f}% | {status}")
        print("-" * 110)
    else:
        print("No NaN values found")

    # 3. Выполнение KNN-импутации
    print("\n--- 3. Выполнение KNN-импутации (n_neighbors=5) ---")
    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed_array = knn_imputer.fit_transform(X_df)

    print(f"Проверка после импутации: осталось NaN = {np.isnan(X_imputed_array).sum()}")

    np.savez_compressed(output_file, data=X_imputed_array, cols=cols, rows=rows)
    return output_file
