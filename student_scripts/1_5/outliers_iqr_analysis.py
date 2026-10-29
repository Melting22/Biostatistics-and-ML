import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Настройка путей на флешке
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")

os.makedirs(FOLDER_1_5, exist_ok=True)

# Загружаем данные (берем knn_imputed из 1_5 или оригинал из 1_4)
INPUT_FILE = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")

print("--- Загрузка данных для анализа IQR ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)
total_patients = len(rows)

# ==========================================
# Поиск выбросов по IQR для каждого признака
# ==========================================
print("--- Расчет межквартильного размаха (IQR) по признакам ---")
patient_broken_counts = {patient_id: 0 for patient_id in rows}
iqr_records = []

for col in cols:
    col_data = pd.to_numeric(X_df[col], errors="coerce")
    
    Q1 = col_data.quantile(0.25)
    Q3 = col_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Границы усов (можно использовать 1.5 или 3.0)
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    
    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
    
    for patient_id, val in outliers.items():
        patient_broken_counts[patient_id] += 1
        iqr_records.append({
            'patient_id': patient_id,
            'feature': col,
            'value': val
        })

df_iqr_all = pd.DataFrame(iqr_records)

# Превращаем в удобный датафрейм по пациентам
df_patient_stats = pd.DataFrame(list(patient_broken_counts.items()), columns=['patient_id', 'broken_features_count'])
df_patient_stats = df_patient_stats.set_index('patient_id')

# Фильтруем пациентов с завышенным числом сломанных параметров (порог поставьте сами)
THRESHOLD = 50
suspect_patients = df_patient_stats[df_patient_stats['broken_features_count'] >= THRESHOLD]

print(f"\n📊 Всего пациентов в выборке: {total_patients}")
print(f"Пациентов с числом выбросов по IQR >= {THRESHOLD}: {len(suspect_patients)}")
print(suspect_patients.sort_values(by='broken_features_count', ascending=False).head(20))

# Сохраняем отчет
iqr_report_path = os.path.join(FOLDER_1_5, f"iqr_patients_with_more_than_{THRESHOLD}_outliers.csv")
suspect_patients.to_csv(iqr_report_path)
print(f"💾 Список подозрительных пациентов сохранен: {iqr_report_path}")


# ==========================================
# Визуализация результатов IQR
# ==========================================
print("--- Генерация графиков для IQR ---")
plt.figure(figsize=(12, 6))
sns.histplot(df_patient_stats['broken_features_count'], bins=30, color='teal', kde=True)
plt.axvline(x=THRESHOLD, color='crimson', linestyle='--', linewidth=2, label=f'Порог отсечки ({THRESHOLD} выбросов)')
plt.title("Распределение количества IQR-выбросов на одного пациента", fontsize=12, fontweight='bold')
plt.xlabel("Количество параметров, вышедших за 'усы'")
plt.ylabel("Количество пациентов")
plt.legend()

iqr_plot_path = os.path.join(FOLDER_1_5, "visualization_iqr_patient_distribution.png")
plt.savefig(iqr_plot_path, dpi=300)
plt.close()
print(f"📊 График распределения сохранен: {iqr_plot_path}")