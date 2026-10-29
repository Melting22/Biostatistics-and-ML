import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Настройка путей на флешке
USB_ROOT = "/media/melting/5A80-1912/student_data"
FOLDER_1_5 = os.path.join(USB_ROOT, "1_5")

os.makedirs(FOLDER_1_5, exist_ok=True)

# Загружаем данные
INPUT_FILE = os.path.join(FOLDER_1_5, "mrt_matrix_knn_imputed.npz")

print("--- Загрузка данных для Isolation Forest ---")
data = np.load(INPUT_FILE, allow_pickle=True)
X = data["data"]
cols = data["cols"]
rows = data["rows"]

X_df = pd.DataFrame(X, columns=cols, index=rows)

# Очищаем от возможных NaN
X_clean = X_df.fillna(X_df.median())

# ==========================================
# 2. ШАГ СТАНДАРТИЗАЦИИ (Критически важно!)
# ==========================================
print("--- Стандартизация признаков (StandardScaler) ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# ==========================================
# 3. Обучение Isolation Forest на масштабированных данных
# ==========================================
print("--- Обучение модели Isolation Forest ---")
iso_forest = IsolationForest(contamination=0.02, random_state=42)
preds = iso_forest.fit_predict(X_scaled)

anomaly_mask = (preds == -1)
anomalous_patients = X_df.index[anomaly_mask].tolist()

print(f"\n📊 Найдено аномальных пациентов: {len(anomalous_patients)}")
print("Список аномальных ID:", anomalous_patients)

# Сохраняем отчет
df_iforest_report = pd.DataFrame({'patient_id': anomalous_patients, 'status': 'Anomalous'})
iforest_report_path = os.path.join(FOLDER_1_5, "isolation_forest_anomalies_list.csv")
df_iforest_report.to_csv(iforest_report_path, index=False)


# ==========================================
# 4. Построение адекватной PCA-карты
# ==========================================
print("--- Генерация читаемой 2D-карты аномалий (PCA) ---")
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_2d, columns=['PC1', 'PC2'], index=rows)
df_pca['Anomalous'] = anomaly_mask

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_pca, x='PC1', y='PC2', 
    hue='Anomalous', palette={False: 'royalblue', True: 'crimson'}, 
    alpha=0.85, s=70
)
plt.title("Многомерные аномалии головного мозга (Isolation Forest + PCA)", fontsize=13, fontweight='bold')
plt.xlabel("Главная компонента 1 (PC1)")
plt.ylabel("Главная компонента 2 (PC2)")
plt.legend(title="Статус пациента", labels=["Норма / Обычный профиль", "Аномалия (Выброс)"])

iforest_plot_path = os.path.join(FOLDER_1_5, "visualization_isolation_forest_pca.png")
plt.savefig(iforest_plot_path, dpi=300)
plt.close()
print(f"📊 Новая читаемая карта аномалий сохранена: {iforest_plot_path}")
print("\n✅ Готово! Перезапустите скрипт и проверьте новый график в папке 1_5.")