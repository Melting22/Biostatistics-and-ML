import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Импортируем пути из вашего модуля config (или задаем вручную)
from config import USB_ROOT, NOSOLOGIES_FILE

# Путь к директории с датасетами
datasets_dir = os.path.join(USB_ROOT, "Datasets")
dataset_files = [
    "Timur_hard_dataset.npz",
    "Timur_soft_dataset.npz",
    "Timur_combat_dataset.npz"
]

# Загружаем словарь с диагнозами (метаданные пациентов)
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_data = json.load(f)

print("🚀 Запуск пайплайна обучения XGBoost для всех датасетов...")

for filename in dataset_files:
    filepath = os.path.join(datasets_dir, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        continue
        
    print(f"\n" + "="*60)
    print(f"📂 ОБРАБОТКА ДАТАСЕТА: {filename}")
    print("="*60)
    
    # 1. Загрузка .npz файла
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    rows = data_load["rows"]
    
    # 2. Собираем целевую переменную (y) для пациентов из датасета
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
    
    # 3. Кодируем текстовые метки классов в числа (0, 1, 2...)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"🔹 Найденные классы (диагнозы): {list(le.classes_)}")
    print(f"🔹 Размерность матрицы признаков X: {X_filtered.shape}")
    print(f"🔹 Размерность целевого вектора y: {y.shape}")
    
    # 4. Стратифицированное разбиение на Train / Test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 5. Инициализация и обучение XGBoost Classifier
    # Параметры подобраны с учетом борьбы с переобучением для малых выборок
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print("⏳ Обучение модели XGBoost...")
    clf.fit(X_train, y_train)
    
    # 6. Предсказание на тестовой выборке
    y_pred = clf.predict(X_test)
    
    # 7. Расчет ключевых метрик качества
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ОЦЕНКИ ДЛЯ {filename}:")
    print(f"  • Balanced Accuracy: {bal_acc:.4f}")
    print(f"  • F1-score (macro):  {f1_macro:.4f}")
    
    print("\n📋 Подробный отчет по классам (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # 8. Построение и сохранение матрицы ошибок (Confusion Matrix)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    dataset_prefix = filename.replace(".npz", "")
    plt.title(f"Confusion Matrix: {dataset_prefix}", fontsize=11, fontweight='bold')
    plt.xlabel("Прогноз модели")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    
    cm_path = os.path.join(datasets_dir, f"{dataset_prefix}_xgb_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"💾 Матрица ошибок сохранена в файл: {cm_path}")

print("\n🎉 Обучение и оценка всех моделей XGBoost успешно завершены!")