import os
import numpy as np
import pandas as pd

# Путь к директории с датасетами
USB_ROOT = "/media/melting/5A80-1912/"
DATASETS_DIR = os.path.join(USB_ROOT, "Datasets")

# Список файлов для проверки
dataset_files = [
    "Timur_hard_dataset.npz",
    "Timur_soft_dataset.npz",
    "Timur_combat_dataset.npz"
]

print("🔍 --- НАЧАЛО ПРОВЕРКИ ДАТАСЕТОВ НА NaN, Inf и 0 --- 🔍\n")

report_list = []

for filename in dataset_files:
    filepath = os.path.join(DATASETS_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️  Файл не найден: {filepath}")
        continue
        
    # Загружаем .npz файл
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    cols = data_load["cols"]
    rows = data_load["rows"]
    
    # 1. Считаем размеры
    n_patients, n_features = X.shape
    total_elements = X.size
    
    # 2. Подсчет аномальных / пропущенных значений
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    n_zeros = (X == 0).sum()
    
    # 3. Проценты от общего количества ячеек
    pct_nan = (n_nan / total_elements) * 100
    pct_zeros = (n_zeros / total_elements) * 100
    
    # Заносим результаты в список
    report_list.append({
        "Датасет": filename,
        "Пациенты (строки)": n_patients,
        "Признаки (столбцы)": n_features,
        "Всего ячеек": total_elements,
        "Кол-во NaN": n_nan,
        "% NaN": f"{pct_nan:.2f}%",
        "Кол-во Inf": n_inf,
        "Кол-во 0": n_zeros,
        "% Нулей": f"{pct_zeros:.2f}%"
    })

# Формируем и красиво выводим итоговую таблицу
df_report = pd.DataFrame(report_list)

print("📊 --- СВОДНЫЙ ОТЧЕТ ПРОВЕРКИ --- 📊")
print(df_report.to_string(index=False))

print("\n--- ДЕТАЛИЗАЦИЯ И РЕКОМЕНДАЦИИ ---")
for report in report_list:
    fname = report["Датасет"]
    n_nan = report["Кол-во NaN"]
    n_inf = report["Кол-во Inf"]
    n_zeros = report["Кол-во 0"]
    
    status = []
    if n_nan == 0 and n_inf == 0:
        status.append("✅ Нет пропусков (NaN/Inf)")
    else:
        status.append(f"❌ НАЙДЕНЫ ПРОПУСКИ: NaN={n_nan}, Inf={n_inf}")
        
    if n_zeros > 0:
        status.append(f"ℹ️ Содержит {n_zeros} нулей ({report['% Нулей']})")
    else:
        status.append("✅ Точных нулей нет")
        
    print(f"\n🔹 {fname}:")
    for stat in status:
        print(f"   • {stat}")

print("\n🎉 Проверка завершена!")