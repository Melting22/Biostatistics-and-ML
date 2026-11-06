import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from config import DIR_3_1, EXCEL_REPORT, ANOMALIES_CSV

def generate_hard_dataset(base_npz_path):
    print("\n--- ЭТАП 2: Генерация Timur_hard_dataset.npz ---")
    
    data_load = np.load(base_npz_path, allow_pickle=True)
    X_df = pd.DataFrame(data_load["data"], columns=data_load["cols"], index=data_load["rows"])
    rows_raw = data_load["rows"]

    # Отчет жесткого отбора
    df_drop_report = pd.read_excel(EXCEL_REPORT, sheet_name=0, header=6)
    df_drop_report.columns = [str(c).strip() for c in df_drop_report.columns]
    hard_drop_list = df_drop_report[df_drop_report['Приоритет'].astype(str).str.contains('Жёстк', na=False)]['Признак'].tolist()

    # Аудит аномалий (фильтр пациентов <= 200)
    df_anom = pd.read_csv(ANOMALIES_CSV)
    anom_col = [c for c in df_anom.columns if 'Z' in c or 'IQR' in c or 'Экстремальных' in c][0]
    patient_anomalies = dict(zip(df_anom['patient_id'], df_anom[anom_col]))

    valid_hard_patients = [p for p in rows_raw if patient_anomalies.get(p, 0) <= 200]
    cols_to_drop = [c for c in hard_drop_list if c in X_df.columns]

    X_hard = X_df.loc[valid_hard_patients].drop(columns=cols_to_drop, errors='ignore')

    # Импутация единичных нулей (<=5%)
    X_hard = X_hard.replace(0, np.nan)
    if X_hard.isna().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_hard[:] = imputer.fit_transform(X_hard)

    hard_path = os.path.join(DIR_3_1, "Timur_hard_dataset.npz")
    np.savez(hard_path, data=X_hard.values, cols=X_hard.columns.values, rows=X_hard.index.values)
    print(f"✅ Сохранен: {hard_path} | Форма: {X_hard.shape}")
    return hard_path