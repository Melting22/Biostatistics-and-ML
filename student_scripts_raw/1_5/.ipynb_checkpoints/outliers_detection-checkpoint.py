import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# 1. Настройка путей на флешке
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")

os.makedirs(FOLDER_1_5, exist_ok=True)

# Берем файл после импутации или фильтрации (например, knn_imputed или filtered_no_imputation)
INPUT_FILE = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")

print("--- Загрузка данных для поиска выбросов ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)
total_patients = len(rows)
print(f"Форма данных: {X_df.shape}, Пациентов: {total_patients}\n")


# ==========================================
# ЧАСТЬ 1: Метод IQR (Межквартильный размах)
# ==========================================
print("--- 1. Поиск выбросов методом IQR (одномерные аномалии) ---")
iqr_outliers_records = []

# Множество для сбора уникальных ID пациентов-нарушителей по IQR
iqr_suspicious_patients = set()

for col in cols:
    col_data = pd.to_numeric(X_df[col], errors="coerce")
    
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Границы усов (обычно 1.5 * IQR или 3.0 * IQR для жестких выбросов)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Находим пациентов, вышедших за усы
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    
    for patient_id, val in outliers.items():
        iqr_suspicious_patients.add(patient_id)
        iqr_outliers_records.append({
            'patient_id': patient_id,
            'feature': col,
            'value': val,
            'method': 'IQR'
        })

df_iqr_outliers = pd.DataFrame(iqr_outliers_records)
print(f"Найдено записей-выбросов по IQR: {len(df_iqr_outliers)}")
print(f"Уникальных пациентов с выбросами по IQR: {len(iqr_suspicious_patients)}")


# ==========================================
# ЧАСТЬ 2: Метод Isolation Forest (Многомерные аномалии)
# ==========================================
print("\n--- 2. Поиск выбросов методом Isolation Forest (многомерные аномалии) ---")
# Заполняем NaN нулями или медианой на время работы модели (если остались)
X_clean_for_if = X_df.fillna(X_df.median())

# Инициализируем Isolation Forest (contamination='auto' или задаем процент, например 0.02 - 2%)
iso_forest = IsolationForest(contamination=0.02, random_state=42)
preds = iso_forest.fit_predict(X_clean_for_if)

# isolation forest возвращает -1 для аномалий и 1 для нормальных точек
anomaly_mask = (preds == -1)
iforest_suspicious_patients = X_df.index[anomaly_mask].tolist()

print(f"Найдено аномальных пациентов через Isolation Forest: {len(iforest_suspicious_patients)}")


# ==========================================
# ЧАСТЬ 3: Формирование сводного отчета для аудита
# ==========================================
print("\n--- 3. Сохранение отчетов в папку 1_5 ---")

# Сохраняем детали по IQR
iqr_report_path = os.path.join(FOLDER_1_5, "outliers_iqr_report.csv")
df_iqr_outliers.to_csv(iqr_report_path, index=False)

# Сохраняем список подозрительных пациентов от Isolation Forest
iforest_report_path = os.path.join(FOLDER_1_5, "outliers_isolation_forest.csv")
pd.DataFrame({'patient_id': iforest_suspicious_patients, 'method': 'IsolationForest'}).to_csv(iforest_report_path, index=False)

print(f"💾 Отчет по IQR сохранен: {iqr_report_path}")
print(f"💾 Отчет по Isolation Forest сохранен: {iforest_report_path}")
print("\n✅ Готово! Теперь вы можете открыть эти CSV-файлы и провести ручной аудит (п. 5.d.iii).")