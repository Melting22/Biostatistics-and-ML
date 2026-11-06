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


def show_patient_data(patient_identifier):
  """Функция выводит все параметры пациента по его индексу или коду (ID)."""
  # Проверяем, передан ли индекс (число) или строковый код пациента
  if isinstance(patient_identifier, int):
    idx = patient_identifier
    if idx < 0 or idx >= len(rows):
      print(f"❌ Ошибка: индекс {idx} вне диапазона (от 0 до {len(rows)-1})")
      return
  else:
    # Ищем индекс по строковому коду пациента
    matching_indices = np.where(rows == patient_identifier)[0]
    if len(matching_indices) == 0:
      print(f"❌ Пациент с кодом '{patient_identifier}' не найден.")
      return
    idx = matching_indices[0]

  patient_id = rows[idx]
  print(f"\n🔍 Данные пациента: {patient_id} (индекс в массиве: {idx})")
  print("-" * 65)
  print(f"{'№':<4} | {'Название признака (колонка)':<40} | {'Значение'}")
  print("-" * 65)

  # Выводим все параметры пациента
  for col_idx, col_name in enumerate(cols):
    val = X_df.iloc[idx, col_idx]
    print(f"{col_idx:<4} | {str(col_name):<40} | {val}")

  print("-" * 65)


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
# Вы можете указать числовой индекс (например, первый пациент: 0)
# Или строковый код пациента из вашего списка 5%-й выборки (например, 'Subj_001')
target = 'sub-50004'

# Запускаем вывод
show_patient_data(target)