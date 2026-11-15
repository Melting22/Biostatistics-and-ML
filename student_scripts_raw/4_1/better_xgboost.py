import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from config import USB_ROOT, NOSOLOGIES_FILE

# Путь к директории с датасетами
datasets_dir = os.path.join(USB_ROOT, "Datasets")
dataset_files = [
    "Timur_hard_dataset.npz",
    "Timur_soft_dataset.npz",
    "Timur_combat_dataset.npz"
]

# Загружаем метаданные пациентов
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_data = json.load(f)

print("🚀 Запуск улучшенного пайплайна XGBoost с балансировкой классов...")

for filename in dataset_files:
    filepath = os.path.join(datasets_dir, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        continue
        
    print(f"\n" + "="*60)
    print(f"📂 ОБРАБОТКА ДАТАСЕТА (С БАЛАНСИРОВКОЙ): {filename}")
    print("="*60)
    
    # 1. Загрузка данных
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    rows = data_load["rows"]
    
    # 2. Извлекаем целевую переменную
    y_raw = []
    valid_indices = []
    
    for idx, pid in enumerate(rows):
        p_info = nosologies_data.get(pid, {})
        disease = p_info.get('nosology', None)
        if disease:
            y_raw.append(disease)
            valid_indices.append(idx)
            
    X_filtered = X[valid_indices]
    y_raw = np.array(y_raw)
    
    # 3. Кодирование меток
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"🔹 Классы: {list(le.classes_)}")
    print(f"🔹 Размерность матрицы X: {X_filtered.shape}")
    
    # 4. Стратифицированное разбиение Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 5. РАСЧЕТ ВЕСОВ КЛАССОВ (Борьба с перекосом в сторону CONTROL)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(zip(np.unique(y_train), class_weights))
    
    # Создаем персональный вес для каждого пациента в обучающей выборке
    sample_weights = np.array([class_weights_dict[label] for label in y_train])
    print(f"⚖️ Рассчитанные веса классов: {dict(zip(le.classes_, np.round(class_weights, 2)))}")
    
    # 6. Инициализация XGBoost
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print("⏳ Обучение модели с учетом весов классов...")
    # Передаем веса через параметр sample_weight в метод fit
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    # 7. Предсказание
    y_pred = clf.predict(X_test)
    
    # 8. Оценка по ключевым метрикам проекта
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 УЛУЧШЕННЫЕ РЕЗУЛЬТАТЫ ДЛЯ {filename}:")
    print(f"  • Balanced Accuracy: {bal_acc:.4f} (Требование этапа 4: оценка дисбаланса)")
    print(f"  • F1-score (macro):  {f1_macro:.4f}[cite: 12, 13]")
    
    print("\n📋 Подробный отчет по классам:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # 9. Сохранение скорректированной матрицы ошибок
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Purples', 
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    dataset_prefix = filename.replace(".npz", "")
    plt.title(f"Balanced Confusion Matrix: {dataset_prefix}", fontsize=11, fontweight='bold')
    plt.xlabel("Прогноз модели")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    
    cm_path = os.path.join(datasets_dir, f"{dataset_prefix}_xgb_balanced_cm.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"💾 Сбалансированная матрица ошибок сохранена: {cm_path}")

print("\n🎉 Обучение с балансировкой классов успешно завершено!")