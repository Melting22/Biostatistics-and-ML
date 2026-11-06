import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# 1. Пути к директориям (читаем из 1_4, сохраняем результат только в 1_5)
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_4 = os.path.join(USB_ROOT, "1_4/a1")
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")

os.makedirs(FOLDER_1_5, exist_ok=True)

# Исходный файл
INPUT_FILE = os.path.join(FOLDER_1_4, "mrt_matrix_filtered_no_imputation.npz")

print("--- 1. Загрузка данных из папки 1_4 (только чтение) ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)
total_patients = len(rows)

print(f"Форма загруженного массива: {X_df.shape}")
print(f"📊 Всего пациентов в выборке: {total_patients}\n")


# 2. Анализ и вывод таблицы процентного соотношения NaN по столбцам
nan_stats = []
threshold_5_pct = total_patients * 0.05

for col_idx, col_name in enumerate(cols):
  # Исправлено: обращаемся к столбцу по его имени (col_name) или через iloc[:, col_idx]
  col_data = pd.to_numeric(X_df[col_name], errors="coerce")
  nan_count = col_data.isna().sum()
  nan_pct = (nan_count / total_patients) * 100

  if nan_count > 0:
    status = (
        "🟢 Допустимо для импутации (<= 5%)"
        if nan_count <= threshold_5_pct
        else "🔴 Превышает 5%!"
    )
    nan_stats.append((col_idx, col_name, nan_count, nan_pct, status))

print("--- 📉 ТАБЛИЦА ПРОЦЕНТНОГО СОДЕРЖАНИЯ NaN ПО СТОЛБЦАМ ---")
if len(nan_stats) > 0:
  print(f"Найдено столбцов с пропусками (NaN): {len(nan_stats)}")
  print("-" * 110)
  print(
      f"{'Индекс':<8} | {'Название признака':<50} | {'Кол-во NaN':<12} |"
      f" {'Процент (%)':<12} | {'Статус'}"
  )
  print("-" * 110)
  for idx, name, count, pct, status in nan_stats:
    print(
        f"{idx:<8} | {str(name):<50} | {count:<12} | {pct:<11.2f}% | {status}"
    )
  print("-" * 110)
else:
  print("✅ Пропусков (NaN) ни в одном столбце не обнаружено.")


# 3. Выполнение KNN-импутации
print("\n--- 3. Выполнение KNN-импутации (n_neighbors=5) ---")
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_array = knn_imputer.fit_transform(X_df)

print(
    f"Проверка после импутации: осталось NaN ="
    f" {np.isnan(X_imputed_array).sum()}"
)


# 4. Сохранение ТОЛЬКО NPZ-файла в папку 1_5
output_npz_path = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")

np.savez_compressed(
    output_npz_path, data=X_imputed_array, cols=cols, rows=rows
)

print(f"\n💾 Заполненный NPZ-тензор успешно сохранен в 1_5: {output_npz_path}")
print("✅ Исходная папка 1_4 осталась в абсолютной сохранности!")