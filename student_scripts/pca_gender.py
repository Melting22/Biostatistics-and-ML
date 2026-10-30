import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Настройка путей
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")
NOSOLOGIES_FILE = "/media/melting/5A80-1912/student_data/1_3/nosologies.json"  # Путь к файлу с метаданными

# Загружаем МРТ-данные
INPUT_FILE = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")
print("--- Загрузка данных и метаданных ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)
X_clean = X_df.fillna(X_df.median())

# Загружаем nosologies.json для получения пола пациентов
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_dict = json.load(f)

# Извлекаем пол для каждого пациента из матрицы rows
genders = []
for patient_id in rows:
    # Убираем или оставляем префикс sub- в зависимости от ключей в json
    # В nosologies.py ключи обычно имеют вид 'sub-...'
    p_data = nosologies_dict.get(patient_id)
    gender = "Unknown"
    if p_data:
        # Проверяем поле gender, которое проставляется в nosologies.py
        gender = p_data.get('gender', 'Unknown')
    genders.append(gender)

# 2. Стандартизация и PCA
print("--- Стандартизация и расчет PCA ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_2d, columns=['PC1', 'PC2'], index=rows)
df_pca['Gender'] = genders

# 3. Визуализация (исправленная палитра и цвета)
plt.figure(figsize=(10, 8))

# Добавим колонку с понятным отображением для легенды, чтобы Seaborn не путался
gender_map = {'M': 'Мужчины (M)', 'F': 'Женщины (F)', 'O': 'Другие', 'Unknown': 'Не указано'}
df_pca['Gender_Label'] = df_pca['Gender'].map(gender_map)

sns.scatterplot(
    data=df_pca, x='PC1', y='PC2', 
    hue='Gender_Label', 
    palette={
        'Мужчины (M)': 'royalblue', 
        'Женщины (F)': 'deeppink', 
        'Другие': 'gray', 
        'Не указано': 'black'
    }, 
    alpha=0.8, s=60
)

plt.title("Распределение пациентов на PCA по полу (Gender)", fontsize=13, fontweight='bold')
plt.xlabel("Главная компонента 1 (PC1)")
plt.ylabel("Главная компонента 2 (PC2)")
plt.legend(title="Пол")

output_path = os.path.join(FOLDER_1_5, "visualization_pca_by_gender.png")
pltفا = plt.savefig(output_path, dpi=300)
plt.close()
print(f"📊 Исправленный график по полу сохранен: {output_path}")