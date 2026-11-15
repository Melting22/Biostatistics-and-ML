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
print("🚀 ФИНАЛЬНЫЙ ПАЙПЛАЙН: Все болезни + Скорректированный порог уверенности")
print("="*65)

filepath = os.path.join(datasets_dir, "Timur_combat_dataset.npz")
if not os.path.exists(filepath):
    print(f"⚠️ Файл не найден: {filepath}")
else:
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    rows = data_load["rows"]
    
    # Собираем все классы
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
    
    print(f"🔹 Доступные классы: {list(le.classes_)}")
    print(f"🔹 Размерность матрицы признаков: {X_filtered.shape}")
    
    # Стратифицированное разбиение (80/20)[cite: 10, 12]
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Обучаем CatBoost с балансировкой
    clf = CatBoostClassifier(
        iterations=400, learning_rate=0.03, depth=4,
        auto_class_weights='Balanced', random_seed=42, verbose=False
    )
    clf.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)
    
    # Получаем вероятности
    probabilities = clf.predict_proba(X_test)
    max_probs = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    
    # Скорректированный порог уверенности (0.28 вместо 0.45)
    CONFIDENCE_THRESHOLD = 0.28
    filtered_preds = []
    
    for pred, prob in zip(predictions, max_probs):
        if prob >= CONFIDENCE_THRESHOLD:
            filtered_preds.append(pred)
        else:
            filtered_preds.append(-1) # Отсекаем сомнительные
            
    valid_mask = np.array(filtered_preds) != -1
    total_test = len(y_test)
    retained_test = np.sum(valid_mask)
    
    print(f"\n📊 АНАЛИЗ ПОРОГА УВЕРЕННОСТИ (p >= {CONFIDENCE_THRESHOLD}):")
    print(f"  • Всего тестовых пациентов: {total_test}")
    print(f"  • Прошло фильтр (уверенные прогнозы): {retained_test} ({retained_test/total_test*100:.1f}%)")
    print(f"  • Отсечено сомнительных прогнозов: {total_test - retained_test}")
    
    if retained_test > 0:
        y_test_fin = y_test[valid_mask]
        y_pred_fin = np.array(filtered_preds)[valid_mask]
        
        bal_acc = balanced_accuracy_score(y_test_fin, y_pred_fin)
        f1_macro = f1_score(y_test_fin, y_pred_fin, average='macro')
        
        print(f"\n🏆 МЕТРИКИ КАЧЕСТВА (по уверенным прогнозам):")
        print(f"  • Balanced Accuracy: {bal_acc:.4f}[cite: 12, 13]")
        print(f"  • F1-score (macro):  {f1_macro:.4f}[cite: 12, 13]")
        
        print("\n📋 Подробный отчет по классам:")
        print(classification_report(y_test_fin, y_pred_fin, target_names=le.classes_, zero_division=0))
        
        # Строим и сохраняем красивую матрицу ошибок
        plt.figure(figsize=(9, 7))
        cm = confusion_matrix(y_test_fin, y_pred_fin, labels=np.arange(len(le.classes_)))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=le.classes_, yticklabels=le.classes_
        )
        plt.title(f"Optimized Multiclass CM (p >= {CONFIDENCE_THRESHOLD}): Timur_combat_dataset", fontsize=11, fontweight='bold')
        plt.xlabel("Прогноз модели (Уверенный)")
        plt.ylabel("Истинный класс")
        plt.tight_layout()
        
        final_cm_path = os.path.join(plots_dir, "Timur_combat_dataset_final_optimized_cm.png")
        plt.savefig(final_cm_path, dpi=300)
        plt.close()
        print(f"\n💾 Финальная матрица ошибок успешно сохранена в: {final_cm_path}")