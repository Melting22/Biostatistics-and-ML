import numpy as np
import pandas as pd

# Укажите путь к вашему файлу данных на флешке
DATA_FILE = (
    "/media/melting/5A80-1912/student_data/measurements_with_legend.npz"
)

# Загружаем данные
data = np.load(DATA_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X)

print(
    "--- ЭТАП 1, ПУНКТ 4b: Проверка аномалий (недопустимые минусы и нули"
    " по признакам) ---"
)

invalid_negatives = []

for col_idx, col_name in enumerate(cols):
  col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
  col_name_str = str(col_name).lower()

  # Проверяем на отрицательные значения (исключая легитимные /Min и кривизну)
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
      invalid_negatives.append((col_name, neg_count))

print(
    f"Обнаружено признаков с недопустимыми отрицательными значениями:"
    f" {len(invalid_negatives)}"
)
if len(invalid_negatives) > 0:
  for name, count in invalid_negatives[:10]:
    print(f"  - {name}: {count} отрицательных значений")
else:
  print("✅ Недопустимых отрицательных значений не обнаружено.")


# 🔍 Поиск признаков с нулевыми значениями и вывод в виде компактной таблицы
print(
    "\n--- АНАЛИЗ ПРИЗНАКОВ С НУЛЕВЫМИ ЗНАЧЕНИЯМИ (0) ---"
)
features_with_zeros = []

for col_idx, col_name in enumerate(cols):
  col_data = pd.to_numeric(X_df[col_idx], errors="coerce")
  zero_count = (col_data == 0).sum()
  if zero_count > 0:
    features_with_zeros.append((col_idx, col_name, zero_count))

if len(features_with_zeros) > 0:
  print(f"Найдено признаков с нулями: {len(features_with_zeros)}")
  print("-" * 90)
  print(f"{'№':<4} | {'Название признака':<70} | {'Кол-во нулей'}")
  print("-" * 90)

  for col_idx, col_name, zero_count in features_with_zeros:
    print(f"{col_idx:<8} | {str(col_name):<70} | {zero_count}")
  print("-" * 90)
else:
  print("✅ Признаков с нулевыми значениями в данных не обнаружено.")


print(
    "\n--- ЭТАП 1, ПУНКТ 4c: Выборочная ручная сверка 5% записей с Excel ---"
)
total_patients = len(rows)
sample_size = max(1, int(total_patients * 0.05))

np.random.seed(42)
random_indices = np.random.choice(
    total_patients, size=sample_size, replace=False
)

print(
    f"Отобрано {sample_size} пациентов для проверки (5% от {total_patients}"
    " общей выборки):"
)
print("-" * 65)
print(f"{'№':<4} | {'Код пациента (Rows)':<20} | {'Индекс в массиве'}")
print("-" * 65)

for i, idx in enumerate(random_indices, 1):
  patient_id = rows[idx]
  print(f"{i:<4} | {str(patient_id):<20} | {idx}")

print("-" * 65)
print(
    "💡 Инструкция: найдите этих пациентов по их кодам в оригинальных таблицах"
    " Excel"
    "\n   на флешке и выборочно сопоставьте их параметры для контроля"
    " качества."
)