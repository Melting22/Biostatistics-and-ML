import os
import numpy as np
import pandas as pd
from config import MEASUREMENTS_FILE, ZEROS_EXCEL, DIR_3_1

def run_zero_cleaning():
    print("\n--- ЭТАП 1: Первичная очистка признаков с >5% нулей ---")
    
    # Загружаем исходные данные
    data_load = np.load(MEASUREMENTS_FILE, allow_pickle=True)
    X = data_load["data"]
    cols = data_load["cols"]
    rows = data_load["rows"]
    X_df = pd.DataFrame(X, columns=cols, index=rows)
    print(f"Исходная форма матрицы: {X_df.shape}")

    # Читаем черный список из файла аудита нулей
    df_blacklist = pd.read_excel(ZEROS_EXCEL, sheet_name="Больше 5% нулей", skiprows=2)
    df_blacklist = df_blacklist.dropna(subset=["Название признака"])
    blacklist_features = df_blacklist["Название признака"].tolist()

    # Исключаем разреженные признаки
    cols_to_keep = [col for col in cols if col not in blacklist_features]
    X_cleaned = X_df[cols_to_keep].copy()
    
    print(f"✅ Удалено признаков с >5% нулей: {len(blacklist_features)}")
    print(f"✅ Осталось признаков: {X_cleaned.shape[1]}")
    
    # Сохраняем промежуточный результат для дальнейших шагов
    cleaned_npz_path = os.path.join(DIR_3_1, "mrt_matrix_cleaned_base.npz")
    np.savez(cleaned_npz_path, data=X_cleaned.values, cols=np.array(cols_to_keep), rows=rows)
    print(f"💾 Базовый очищенный файл сохранен: {cleaned_npz_path}")
    return cleaned_npz_path

if __name__ == "__main__":
    run_zero_cleaning()