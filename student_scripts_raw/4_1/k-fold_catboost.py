import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
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
print("🔄 ЗАПУСК СТРАТИФИЦИРОВАННОЙ КРОСС-ВАЛИДАЦИИ (Stratified K-Fold CV)")
print("="*65)

filepath = os.path.join(datasets_dir, "Timur_combat_dataset.npz")
if not os.path.exists(filepath):
    print(f"⚠️ Файл не найден: {filepath}")
else:
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    rows = data_load["rows"]
    
    # Собираем все доступные классы
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
    
    # Кодируем метки
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"🔹 Классы для кросс-валидации: {list(le.classes_)}")
    print(f"🔹 Общий размер матрицы признаков: {X_filtered.shape}")
    
    # Настраиваем 5-фолдовую кросс-валидацию со стратификацией
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_balanced_accuracies = []
    fold_f1_macros = []
    
    # Аккумулятор для общей матрицы ошибок по всем фолдам
    total_cm = np.zeros((len(le.classes_), len(le.classes_)), dtype=int)
    
    print(f"\n⏳ Выполнение {n_splits}-кратной кросс-валидации...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_filtered, y), 1):
        X_train, X_test = X_filtered[train_idx], X_filtered[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Инициализируем CatBoost с авто-балансировкой классов
        clf = CatBoostClassifier(
            iterations=300, 
            learning_rate=0.03, 
            depth=4,
            auto_class_weights='Balanced', 
            random_seed=42 + fold, 
            verbose=False
        )
        
        # Обучаем модель на текущем фолде
        clf.fit(X_train, y_train, verbose=False)
        
        # Предсказания на тестовой части фолда
        y_pred = clf.predict(X_test).flatten()
        
        # Считаем метрики
        b_acc = balanced_accuracy_score(y_test, y_pred)
        f1_m = f1_score(y_test, y_pred, average='macro')
        
        fold_balanced_accuracies.append(b_acc)
        fold_f1_macros.append(f1_m)
        
        # Суммируем матрицу ошибок
        total_cm += confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
        
        print(f"  • Фол д {fold}/{n_splits} | Balanced Accuracy: {b_acc:.4f} | F1-macro: {f1_m:.4f}")
        
    # Итоговые усредненные результаты
    print("\n" + "="*50)
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ:")
    print("="*50)
    print(f"  • Средняя Balanced Accuracy: {np.mean(fold_balanced_accuracies):.4f} ± {np.std(fold_balanced_accuracies):.4f}")
    print(f"  • Средний F1-score (macro):  {np.mean(fold_f1_macros):.4f} ± {np.std(fold_f1_macros):.4f}")
    
    # Сохраняем общую суммарную матрицу ошибок кросс-валидации
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        total_cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=le.classes_, yticklabels=le.classes_
    )
    plt.title(f"Aggregated CV Confusion Matrix ({n_splits}-Folds): Timur_combat_dataset", fontsize=11, fontweight='bold')
    plt.xlabel("Прогноз модели")
    plt.ylabel("Истинный класс")
    plt.tight_layout()
    
    cv_cm_path = os.path.join(plots_dir, "Timur_combat_dataset_cv_aggregated_cm.png")
    plt.savefig(cv_cm_path, dpi=300)
    plt.close()
    print(f"\n💾 Суммарная матрица ошибок кросс-валидации сохранена в: {cv_cm_path}")