import numpy as np
from constants import DATA_FILE

data = np.load(DATA_FILE)

# Выводим доступные ключи, чтобы убедиться в правильности названий (как у вас в примере)
print('Доступные ключи в файле:', data.files)

rows = data['rows']

print(f'Общее количество записей (пациентов): {len(rows)}')

# Ищем дубликаты
unique_rows, counts = np.unique(rows, return_counts=True)
duplicates = unique_rows[counts > 1]

if len(duplicates) > 0:
  print(
      f'⚠️ Внимание! Обнаружены дубликаты кодов пациентов в количестве:'
      f' {len(duplicates)}'
  )
  print('Список повторяющихся кодов:', duplicates)

  # Если нужно посмотреть, сколько раз каждый из них повторяется:
  for dup in duplicates:
    count = counts[unique_rows == dup][0]
    print(f'Код пациента "{dup}" встречается {count} раза(в).')
else:
  print(
      '✅ Дубликаты не обнаружены. Все коды пациентов уникальны (критерий Этапа'
      ' 4а выполнен).'
  )