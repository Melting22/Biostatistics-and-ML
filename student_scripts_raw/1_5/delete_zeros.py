import os
import numpy as np
import pandas as pd

# 1. Указываем пути к папкам на флешке
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_4 = os.path.join(USB_ROOT, "1_4/a")
FOLDER_1_5 = os.path.join(USB_ROOT, "1_4/a1")

# Пути к файлам
DATA_FILE = os.path.join(FOLDER_1_4, "measurements_with_legend.npz")
EXCEL_FILE = os.path.join(FOLDER_1_4, "Анализ_нулевых_значений_МРТ.xlsx")

print("--- Загрузка оригинальных данных из папки 1_4 (только чтение) ---")
data = np.load(DATA_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df_original = pd.DataFrame(X, columns=cols, index=rows)
print(f"Исходная размерность матрицы: {X_df_original.shape}")


print("\n--- Загрузка черного списка признаков (>5% нулей) из Excel ---")
df_blacklist_source = pd.read_excel(EXCEL_FILE, sheet_name="Больше 5% нулей", skiprows=2)
df_blacklist_source = df_blacklist_source.dropna(subset=["Индекс"])
blacklist_features = df_blacklist_source["Название признака"].tolist()

print(f"Количество признаков в черном списке для исключения: {len(blacklist_features)}")


print("\n--- Фильтрация признаков (удаление >5% нулей без импутации) ---")
# Оставляем только те столбцы, которых НЕТ в черном списке
cols_to_keep = [col for col in cols if col not in blacklist_features]
X_filtered = X_df_original[cols_to_keep].copy()

print(f"Колонок до фильтрации: {X_df_original.shape[1]}")
print(f"Колонок после исключения разреженных признаков: {X_filtered.shape[1]}")


print("\n--- Сохранение результатов в папку 1_5 на флешке ---")
output_npz_path = os.path.join(FOLDER_1_5, "mrt_matrix_filtered_no_imputation.npz")

# Сохраняем в NPZ-тензор
np.savez(
    output_npz_path,
    data=X_filtered.values,
    cols=np.array(cols_to_keep),
    rows=rows
)

print(f"💾 Отфильтрованный NPZ сохранен в 1_5: {output_npz_path}")
print("\n✅ Готово! Исходники в 1_4 не затронуты, а очищенный массив без импутации лежит в папке 1_4_a1.")