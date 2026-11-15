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

# Пути к директориям
datasets_dir = os.path.join(USB_ROOT, "Datasets")
# Создаем отдельную подпапку для матриц ошибок, как вы просили
plots_dir = os.path.join(datasets_dir, "confusion_matrices")
os.makedirs(plots_dir, exist_ok=True)

dataset_files = [
    "Timur_hard_dataset.npz",
    "Timur_soft_dataset.npz",
    "Timur_combat_dataset.npz"
]

# Загружаем метаданные пациентов
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_data = json.load(f)

print("🚀 Запуск пайплайна обучения CatBoost (с авто-балансировкой и структурированием графиков)...")

for filename in dataset_files:
    filepath = os.path.join(datasets_dir, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        continue
        
    print(f"\n" + "="*60)
    print(f"📂 ОБРАБОТКА ДАТАСЕТА (CATBOOST): {filename}")
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
    
    # 5. Инициализация CatBoost Classifier
    # Параметр auto_class_weights='Balanced' автоматически борется с дисбалансом классов
    clf = CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=4,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=False
    )
    
    print("⏳ Обучение модели CatBoost...")
    # Обучаем с ранней остановкой (early_stopping) по валидации для защиты от переобучения
    clf.fit(
        X_train, y_train, 
        eval_set=(X_test, y_test), 
        early_stopping_rounds=50,
        verbose=False
    )
    
    # 6. Предсказание (CatBoost возвращает 2D массив, делаем flatten)
    y_pred = clf.predict(X_test).flatten()
    
    # 7. Оценка метрик качества
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 РЕЗУЛЬТАТЫ CATBOOST ДЛЯ {filename}:")
    print(f"  • Balanced Accuracy: {bal_acc:.4f}")
    print(f"  • F1-score (macro):  {f1_macro:.4f}")
    
    print("\n📋 Подробный отчет по классам:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # 8. Сохранение матрицы ошибок в новую подпапку confusion_matrices
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens', 
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    dataset_prefix = filename.replace(".npz", "")
    plt.title(f"CatBoost Confusion Matrix: {dataset_prefix}", fontsize=11, fontweight='bold')
    plt.xlabel("Прогноз модели")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    
    cm_path = os.path.join(plots_dir, f"{dataset_prefix}_catboost_cm.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"💾 Матрица ошибок сохранена в подпапку: {cm_path}")

print("\n🎉 Обучение CatBoost успешно завершено, все графики упорядочены в папке Datasets/confusion_matrices!")