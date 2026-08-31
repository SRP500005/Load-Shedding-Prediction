from src.data_loader import load_data
from src.data_validation import validate_data
from src.data_cleaning import clean_data


# Dataset ka path
DATA_PATH = "data/raw/grid_timeseries_2020_2024.csv"


# 1. Dataset load karo
df = load_data(DATA_PATH)

print("Dataset Loaded Successfully")


# 2. First 5 rows dekho
print("\nFirst 5 Rows:")
print(df.head())


# 3. Original dataset validate karo
validate_data(df)


# 4. Dataset clean karo
cleaned_df = clean_data(df)


# 5. Cleaning ke baad result check karo
print("\n===== AFTER DATA CLEANING =====")

print("\nShape:")
print(cleaned_df.shape)

print("\nRemaining Missing Values:")
print(cleaned_df.isnull().sum().sum())

print("\nColumns after cleaning:")
print(cleaned_df.columns.tolist())