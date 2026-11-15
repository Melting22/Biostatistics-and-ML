import os

# Глобальные пути проекта
USB_ROOT = "/media/melting/5A80-1912/"
DIR_1_3 = os.path.join(USB_ROOT, "master_data")
DIR_3_1 = os.path.join(USB_ROOT, "Datasets")
os.makedirs(DIR_3_1, exist_ok=True)

NOSOLOGIES_FILE = os.path.join(DIR_1_3, "nosologies.json")
MEASUREMENTS_FILE = os.path.join(DIR_1_3, "measurements_with_legend.npz")
EXCEL_REPORT = "feature_selection_report1.xlsx"
ANOMALIES_CSV = "master_anomalies_audit_clean.csv"
ZEROS_EXCEL = "Анализ_нулевых_значений_МРТ.xlsx"