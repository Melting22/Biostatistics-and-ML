import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

from config import USB_ROOT, NOSOLOGIES_FILE

# Пути
datasets_dir = os.path.join(USB_ROOT, "Datasets")
plots_dir = os.path.join(datasets_dir, "confusion_matrices")
os.makedirs(plots_dir, exist_ok=True)

# Загружаем метаданные
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_data = json.load(f)

print("="*65)
print("🔬 ЭКСПЕРИМЕНТ 1: Бинарный срез (CONTROL vs SCHZ) на Combat-датасете")
print("="*65)

filepath = os.path.join(datasets_dir, "Timur_combat_dataset.npz")
data_load = np.load(filepath, allow_pickle=True)
X = data_load["data"]
rows = data_load["rows"]

# Фильтруем пациентов только для CONTROL и SCHZ
y_raw_bin = []
X_bin_list = []
for idx, pid in enumerate(rows):
    p_info = nosologies_data.get(pid, {})
    disease = p_info.get('nosology', None)
    if disease in ['CONTROL', 'SCHZ']:
        y_raw_bin.append(disease)
        X_bin_list.append(X[idx])

X_bin = np.array(X_bin_list)
y_bin = np.array([1 if d == 'SCHZ' else 0 for d in y_raw_bin]) # 1 - Шизофрения, 0 - Норма

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bin, y_bin, test_size=0.20, random_state=42, stratify=y_bin
)

clf_bin = CatBoostClassifier(
    iterations=300, learning_rate=0.03, depth=3, 
    auto_class_weights='Balanced', random_seed=42, verbose=False
)
clf_bin.fit(X_train_b, y_train_b)

y_pred_b = clf_bin.predict(X_test_b)
print(f"📊 Результат бинарной классификации (CONTROL vs SCHZ):")
print(classification_report(y_test_b, y_pred_b, target_names=['CONTROL', 'SCHZ']))


print("\n" + "="*65)
print("🎯 ЭКСПЕРИМЕНТ 2: Многоклассовая классификация + Тюнинг порога уверенности")
print("="*65)

# Собираем все классы обратно
y_raw_multi = []
valid_indices = []
for idx, pid in enumerate(rows):
    p_info = nosologies_data.get(pid, {})
    disease = p_info.get('nosology', None)
    if disease:
        y_raw_multi.append(disease)
        valid_indices.append(idx)

X_multi = X[valid_indices]
y_raw_multi = np.array(y_raw_multi)

le = LabelEncoder()
y_multi = le.fit_transform(y_raw_multi)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.20, random_state=42, stratify=y_multi
)

clf_multi = CatBoostClassifier(
    iterations=400, learning_rate=0.03, depth=4, 
    auto_class_weights='Balanced', random_seed=42, verbose=False
)
clf_multi.fit(X_train_m, y_train_m)

# Получаем вероятности для каждого класса (а не просто жесткий класс)
probabilities = clf_multi.predict_proba(X_test_m)
max_probs = np.max(probabilities, axis=1) # Максимальная уверенность модели для каждого пациента
predictions = np.argmax(probabilities, axis=1)

# Тюнинг порога уверенности: отсекаем сомнительные прогнозы (например, если уверенность < 0.45)
CONFIDENCE_THRESHOLD = 0.45
filtered_preds = []
uncertain_count = 0

for pred, prob in zip(predictions, max_probs):
    if prob >= CONFIDENCE_THRESHOLD:
        filtered_preds.append(pred)
    else:
        filtered_preds.append(-1) # Метка неопределенности

print(f"🔍 Всего тестовых пациентов: {len(y_test_m)}")
print(f"⚠️ Пациентов с низкой уверенностью модели (< {CONFIDENCE_THRESHOLD}): {np.sum(np.array(filtered_preds) == -1)}")

# Строим матрицу ошибок только для уверенных предсказаний
valid_mask = np.array(filtered_preds) != -1
if np.sum(valid_mask) > 0:
    y_test_filtered = y_test_m[valid_mask]
    y_pred_filtered = np.array(filtered_preds)[valid_mask]
    
    bal_acc = balanced_accuracy_score(y_test_filtered, y_pred_filtered)
    print(f"📊 Balanced Accuracy (по отфильтрованным уверенным прогнозам): {bal_acc:.4f}")
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=np.unique(y_test_m))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title(f"Threshold-Tuned CM (p >= {CONFIDENCE_THRESHOLD}): Timur_combat_dataset", fontsize=11, fontweight='bold')
    plt.xlabel("Прогноз модели (Уверенный)")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    
    tuned_cm_path = os.path.join(plots_dir, "Timur_combat_dataset_tuned_cm.png")
    plt.savefig(tuned_cm_path, dpi=300)
    plt.close()
    print(f"💾 Сбалансированная матрица с учетом порога сохранена: {tuned_cm_path}")