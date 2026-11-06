import os
import json
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from neuroCombat import neuroCombat
from config import DIR_3_1, EXCEL_REPORT, ANOMALIES_CSV, NOSOLOGIES_FILE

def generate_soft_and_combat(base_npz_path):
    print("\n--- ЭТАП 3: Генерация Timur_soft_dataset.npz и Timur_combat_dataset.npz ---")
    
    data_load = np.load(base_npz_path, allow_pickle=True)
    X_df = pd.DataFrame(data_load["data"], columns=data_load["cols"], index=data_load["rows"])
    rows_raw = data_load["rows"]

    # Отчет мягкого/жесткого отбора
    df_drop_report = pd.read_excel(EXCEL_REPORT, sheet_name=0, header=6)
    df_drop_report.columns = [str(c).strip() for c in df_drop_report.columns]
    soft_drop_list = df_drop_report[df_drop_report['Приоритет'].astype(str).str.contains('Жёстк|Мягк', na=False)]['Признак'].tolist()

    # Фильтр пациентов <= 100 аномалий
    df_anom = pd.read_csv(ANOMALIES_CSV)
    anom_col = [c for c in df_anom.columns if 'Z' in c or 'IQR' in c or 'Экстремальных' in c][0]
    patient_anomalies = dict(zip(df_anom['patient_id'], df_anom[anom_col]))

    valid_soft_patients = [p for p in rows_raw if patient_anomalies.get(p, 0) <= 100]
    cols_to_drop = [c for c in soft_drop_list if c in X_df.columns]

    X_soft = X_df.loc[valid_soft_patients].drop(columns=cols_to_drop, errors='ignore')

    # Импутация мелких нулей
    X_soft = X_soft.replace(0, np.nan)
    if X_soft.isna().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        X_soft[:] = imputer.fit_transform(X_soft)

    # Сохраняем Soft датасет
    soft_path = os.path.join(DIR_3_1, "Timur_soft_dataset.npz")
    np.savez(soft_path, data=X_soft.values, cols=X_soft.columns.values, rows=X_soft.index.values)
    print(f"✅ Сохранен: {soft_path} | Форма: {X_soft.shape}")

    # --- Применяем neuroCombat для получения Combat датасета ---
    print("\n🛡️ Запуск neuroCombat для устранения эффекта сканера...")
    with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
        nosologies_data = json.load(f)

    combat_rows = X_soft.index.tolist()
    covars_list = []
    for pid in combat_rows:
        p_info = nosologies_data.get(pid, {})
        covars_list.append({
            'site': p_info.get('dataset', 'UCLA'),
            'age': float(p_info.get('age', 30)),
            'is_schz': 1 if p_info.get('nosology', '') == 'SCHZ' else 0
        })
    df_covars = pd.DataFrame(covars_list)

    dat_for_combat = X_soft.T.values
    design_df = pd.DataFrame({
        'site': df_covars['site'].values,
        'age': df_covars['age'].fillna(30).values,
        'is_schz': df_covars['is_schz'].values
    })

    combat_results = neuroCombat(
        dat=dat_for_combat,
        covars=design_df,
        batch_col='site',
        categorical_cols=['is_schz'],
        continuous_cols=['age']
    )

    X_combat = combat_results['data'].T
    combat_path = os.path.join(DIR_3_1, "Timur_combat_dataset.npz")
    np.savez(combat_path, data=X_combat, cols=X_soft.columns.values, rows=combat_rows)
    print(f"✅ Сохранен: {combat_path} | Форма: {X_combat.shape}")
    
    return soft_path, combat_path