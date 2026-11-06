import numpy as np
import pandas as pd

# Укажите путь к вашему файлу данных на флешке
DATA_FILE = (
    "/media/melting/5A80-1912/student_data/1_4/a1/mrt_matrix_filtered_no_imputation.npz"
)

# Загружаем данные
data = np.load(DATA_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X)
total_patients = len(rows)

print(
    "--- ЭТАП 1, ПУНКТ 4b: Проверка аномалий (анализ отрицательных чисел и нулей) ---"
)
print(f"📊 Всего пациентов в выборке: {total_patients}\n")

# ==========================================
# 1. АНАЛИЗ ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ ВО ВСЕХ СТОЛБЦАХ
# ==========================================
negative_stats = []

for col_idx, col_name in enumerate(cols):
  col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
  col_name_str = str(col_name).lower()

  neg_count = (col_data < 0).sum()
  neg_pct = (neg_count / total_patients) * 100

  # Проверка, допустимы ли отрицательные значения по смыслу признака
  is_allowed = (
      "/min" in col_name_str
      or "curv" in col_name_str
      or "кривизн" in col_name_str
      or "asymmetry" in col_name_str
      or "ai" in col_name_str
  )
  
  status = "Допустимо (кривизна/мин/асимметрия)" if is_allowed else "⚠️ Возможная аномалия"

  if neg_count > 0:
    negative_stats.append((col_idx, col_name, neg_count, neg_pct, status))

print(f"--- 📉 ТАБЛИЦА ОТРИЦАТЕЛЬНЫХ ЗНАЧЕНИЙ (во всех столбцах) ---")
if len(negative_stats) > 0:
  print(f"Найдено признаков с отрицательными значениями: {len(negative_stats)}")
  print("-" * 115)
  print(
      f"{'Индекс':<8} | {'Название признака':<55} | {'Кол-во минусов':<14} |"
      f" {'Процент (%)':<12} | {'Статус'}"
  )
  print("-" * 115)
  for col_idx, col_name, neg_count, neg_pct, status in negative_stats:
    print(
        f"{col_idx:<8} | {str(col_name):<55} | {neg_count:<14} |"
        f" {neg_pct:<11.2f}% | {status}"
    )
  print("-" * 115)
else:
  print("✅ Отрицательных значений ни в одном столбце не обнаружено.")