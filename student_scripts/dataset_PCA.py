import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Настройка путей на флешке
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")
NOSOLOGIES_FILE = "/media/melting/5A80-1912/student_data/1_3/nosologies.json"

os.makedirs(FOLDER_1_5, exist_ok=True)

# Загружаем МРТ-данные
INPUT_FILE = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")
print("--- Загрузка данных и метаданных для датасетов ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)
X_clean = X_df.fillna(X_df.median())

# Загружаем nosologies.json для определения датасета (UCLA / TCP)
with open(NOSOLOGIES_FILE, encoding="utf-8") as f:
    nosologies_dict = json.load(f)

datasets = []
for patient_id in rows:
    p_data = nosologies_dict.get(patient_id)
    ds = "Unknown"
    if p_data:
        ds = p_data.get("dataset", "Unknown")
    datasets.append(ds)

X_df["Dataset"] = datasets

# 2. Стандартизация и PCA
print("--- Стандартизация и расчет PCA ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_2d, columns=["PC1", "PC2"], index=rows)
df_pca["Dataset"] = X_df["Dataset"]

# Создаем понятные подписи для легенды
ds_map = {"TCP": "Клиника TCP", "UCLA": "Клиника UCLA", "Unknown": "Не указано"}
df_pca["Dataset_Label"] = df_pca["Dataset"].map(ds_map)

# 3. Визуализация
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="Dataset_Label",
    palette={
        "Клиника TCP": "forestgreen",
        "Клиника UCLA": "darkorange",
        "Не указано": "gray",
    },
    alpha=0.8,
    s=60,
)
plt.title(
    "Распределение пациентов на PCA по датасетам (Эффект сканера)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("Главная компонента 1 (PC1)")
plt.ylabel("Главная компонента 2 (PC2)")
plt.legend(title="Датасет / Сканер")

output_path = os.path.join(FOLDER_1_5, "visualization_pca_by_dataset.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"📊 График по датасетам сохранен: {output_path}")
print("\n✅ Скрипт успешно отработал! Проверьте график в папке 1_5.")
