from src.data_loader import load_data
from src.data_validation import validate_data


DATA_PATH = "data/raw/grid_timeseries_2020_2024.csv"


df = load_data(DATA_PATH)


print("Dataset Loaded Successfully")


print("\nFirst 5 Rows:")
print(df.head())


validate_data(df)