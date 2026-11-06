import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import USB_ROOT, NOSOLOGIES_FILE

# Путь к папке с датасетами
DIR_3_1 = os.path.join(USB_ROOT, "Datasets")

datasets_to_process = [
    "Timur_hard_dataset.npz",
    "Timur_soft_dataset.npz",
    "Timur_combat_dataset.npz"
]

# Загружаем метаданные пациентов
with open(NOSOLOGIES_FILE, 'r', encoding='utf-8') as f:
    nosologies_data = json.load(f)

print("--- Запуск PCA-анализа и визуализации для всех датасетов ---")

for filename in datasets_to_process:
    filepath = os.path.join(DIR_3_1, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        continue
        
    print(f"\n📂 Обработка датасета: {filename}")
    data_load = np.load(filepath, allow_pickle=True)
    X = data_load["data"]
    cols = data_load["cols"]
    rows = data_load["rows"]
    
    X_df = pd.DataFrame(X, columns=cols, index=rows)
    
    # Собираем метаданные для пациентов, присутствующих в текущем датасете
    genders = []
    diseases = []
    sites = []
    
    for pid in rows:
        p_info = nosologies_data.get(pid, {})
        genders.append(p_info.get('gender', 'M'))
        diseases.append(p_info.get('nosology', 'CONTROL'))
        sites.append(p_info.get('dataset', 'UCLA'))
        
    df_meta = pd.DataFrame({
        'gender': genders,
        'disease': diseases,
        'site': sites
    }, index=rows)
    
    # 1. Обязательная стандартизация признаков (выравнивание масштаба)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.fillna(0))
    
    # 2. Запуск PCA (2 главные компоненты)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    df_meta['PC1'] = X_pca[:, 0]
    df_meta['PC2'] = X_pca[:, 1]
    expl_var = pca.explained_variance_ratio_ * 100
    
    dataset_prefix = filename.replace(".npz", "")
    
    # --- График 1: Разметка по нозологическим группам (Диагнозам) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_meta, x='PC1', y='PC2', hue='disease', palette='Set1', alpha=0.85, s=60, edgecolor='w', linewidth=0.5
    )
    plt.title(f"PCA ({dataset_prefix}): Разметка по нозологиям\n(PC1: {expl_var[0]:.1f}%, PC2: {expl_var[1]:.1f}%)", fontsize=11, fontweight='bold')
    plt.xlabel(f"Главная компонента 1 ({expl_var[0]:.1f}%)")
    plt.ylabel(f"Главная компонента 2 ({expl_var[1]:.1f}%)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Диагноз", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plot_path_nos = os.path.join(DIR_3_1, f"{dataset_prefix}_pca_nosology.png")
    plt.savefig(plot_path_nos, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Сохранен график нозологий: {plot_path_nos}")
    
    # --- График 2: Разметка по полу (Gender) ---
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df_meta, x='PC1', y='PC2', hue='gender', palette='Set2', alpha=0.8, s=60
    )
    plt.title(f"PCA ({dataset_prefix}): Разметка по полу", fontsize=11, fontweight='bold')
    plt.xlabel("Главная компонента 1")
    plt.ylabel("Главная компонента 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Пол", loc='upper right')
    plt.tight_layout()
    plot_path_gen = os.path.join(DIR_3_1, f"{dataset_prefix}_pca_gender.png")
    plt.savefig(plot_path_gen, dpi=300)
    plt.close()
    print(f"  ✅ Сохранен график пола: {plot_path_gen}")
    
    # --- График 3: Попарное сравнение CONTROL vs SCHZ ---
    df_binary = df_meta[df_meta['disease'].isin(['CONTROL', 'SCHZ'])]
    if len(df_binary) > 0:
        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=df_binary, x='PC1', y='PC2', hue='disease', 
            palette={'CONTROL': 'forestgreen', 'SCHZ': 'crimson'}, alpha=0.85, s=60
        )
        plt.title(f"PCA ({dataset_prefix}): Control vs Schizophrenia", fontsize=11, fontweight='bold')
        plt.xlabel("Главная компонента 1")
        plt.ylabel("Главная компонента 2")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Группа", loc='upper right')
        plt.tight_layout()
        plot_path_bin = os.path.join(DIR_3_1, f"{dataset_prefix}_pca_control_vs_schz.png")
        plt.savefig(plot_path_bin, dpi=300)
        plt.close()
        print(f"  ✅ Сохранен график Control vs SCHZ: {plot_path_bin}")

print("\n🎉 Все PCA-анализы и графики успешно построены и сохранены в папку Datasets!")