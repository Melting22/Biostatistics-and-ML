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
    "--- ЭТАП 1, ПУНКТ 4b: Проверка аномалий (недопустимые минусы и нули) ---"
)

# 1. Проверка на недопустимые отрицательные значения
invalid_negatives = []
for col_idx, col_name in enumerate(cols):
  col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
  col_name_str = str(col_name).lower()

  if (col_data < 0).any():
    is_allowed_negative = (
        "/min" in col_name_str
        or "curv" in col_name_str
        or "кривизн" in col_name_str
        or "asymmetry" in col_name_str
        or "ai" in col_name_str
    )
    if not is_allowed_negative:
      neg_count = (col_data < 0).sum()
      invalid_negatives.append((col_idx, col_name, neg_count))

print(
    f"Обнаружено признаков с недопустимыми отрицательными значениями:"
    f" {len(invalid_negatives)}"
)
if len(invalid_negatives) > 0:
  print(f"{'Индекс':<8} | {'Название признака':<70} | {'Кол-во минусов'}")
  print("-" * 90)
  for col_idx, name, count in invalid_negatives[:10]:
    print(f"{col_idx:<8} | {str(name):<70} | {count}")
else:
  print("✅ Недопустимых отрицательных значений не обнаружено.")


# 2. Поиск признаков с нулями и разделение по порогу 5%
threshold_5_percent = total_patients * 0.05

zeros_less_5_pct = []
zeros_more_5_pct = []

for col_idx, col_name in enumerate(cols):
  col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
  zero_count = (col_data == 0).sum()

  if zero_count > 0:
    zero_pct = (zero_count / total_patients) * 100
    item = (col_idx, col_name, zero_count, zero_pct)

    if zero_count > threshold_5_percent:
      zeros_more_5_pct.append(item)
    else:
      zeros_less_5_pct.append(item)

print(f"\n📊 Всего пациентов в выборке: {total_patients}")
print(f"Порог 5% от выборки составляет: {threshold_5_percent:.1f} пациентов")


# ТАБЛИЦА 1: Признаки, где нулей МЕНЬШЕ 5%
print(
    "\n--- 🟢 ТАБЛИЦА 1: Признаки с нулями < 5% (кандидаты на импутацию)"
    " ---"
)
if len(zeros_less_5_pct) > 0:
  print(f"Найдено признаков: {len(zeros_less_5_pct)}")
  print("-" * 90)
  print(
      f"{'Индекс':<8} | {'Название признака':<70} | {'Кол-во нулей':<12} |"
      f" {'Процент нулей'}"
  )
  print("-" * 90)
  for col_idx, col_name, zero_count, zero_pct in zeros_less_5_pct:
    print(
        f"{col_idx:<8} | {str(col_name):<70} | {zero_count:<12} |"
        f" {zero_pct:.2f}%"
    )
  print("-" * 90)
else:
  print("✅ Признаков с нулями < 5% не найдено.")


# ТАБЛИЦА 2: Признаки, где нулей БОЛЬШЕ 5%
print(
    "\n--- 🔴 ТАБЛИЦА 2: Признаки с нулями > 5% (критические пропуски /"
    " кандидаты на удаление) ---"
)
if len(zeros_more_5_pct) > 0:
  print(f"Найдено признаков: {len(zeros_more_5_pct)}")
  print("-" * 90)
  print(
      f"{'Индекс':<8} | {'Название признака':<70} | {'Кол-во нулей':<12} |"
      f" {'Процент нулей'}"
  )
  print("-" * 90)
  for col_idx, col_name, zero_count, zero_pct in zeros_more_5_pct:
    print(
        f"{col_idx:<8} | {str(col_name):<70} | {zero_count:<12} |"
        f" {zero_pct:.2f}%"
    )
  print("-" * 95)
else:
  print("✅ Признаков с нулями > 5% не найдено.")