from clean_zeros import run_zero_cleaning
from make_hard_dataset import generate_hard_dataset
from make_soft_and_combat_datasets import generate_soft_and_combat

def main():
    print("🚀 === СТАРТ ГЛАВНОГО КОНВЕЙЕРА ДАТАСЕТОВ TIMUR ===")
    
    # Шаг 1: Первичная очистка от >5% нулей
    cleaned_base_path = run_zero_cleaning()
    
    # Шаг 2: Генерация Hard датасета
    generate_hard_dataset(cleaned_base_path)
    
    # Шаг 3: Генерация Soft и Combat датасетов
    generate_soft_and_combat(cleaned_base_path)
    
    print("\n🎉 ВСЕ ДАТАСЕТЫ УСПЕШНО СОЗДАНЫ И ПРОШЛИ ОЧИСТКУ!")

if __name__ == "__main__":
    main()