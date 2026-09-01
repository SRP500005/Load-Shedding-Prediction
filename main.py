from src.data_loader import load_data
from src.data_validation import validate_data
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.preprocessing import prepare_data, time_based_split
from src.baseline_model import train_baseline_model
from src.logistic_regression_model import train_logistic_regression

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

# 6. Exploratory Data Analysis
perform_eda(cleaned_df)

# 7. Prepare data for machine learning
X, y = prepare_data(cleaned_df)

print("\n===== FEATURE PREPARATION =====")

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())


# 8. Time-based train/test split
X_train, X_test, y_train, y_test = time_based_split(
    X,
    y,
    train_ratio=0.8
)

print("\n===== TRAIN TEST SPLIT =====")

print("\nX_train:")
print(X_train.shape)

print("\nX_test:")
print(X_test.shape)

print("\ny_train:")
print(y_train.shape)

print("\ny_test:")
print(y_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# 9. Baseline Model

accuracy, precision, recall, f1, cm = train_baseline_model(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n===== BASELINE MODEL =====")

print("\nAccuracy:")
print(accuracy)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nConfusion Matrix:")
print(cm)

# 10. Logistic Regression Model

(
    logistic_model,
    lr_accuracy,
    lr_precision,
    lr_recall,
    lr_f1,
    lr_roc_auc,
    lr_cm
) = train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test
)

print("\n===== LOGISTIC REGRESSION =====")

print("\nAccuracy:")
print(lr_accuracy)

print("\nPrecision:")
print(lr_precision)

print("\nRecall:")
print(lr_recall)

print("\nF1 Score:")
print(lr_f1)

print("\nROC-AUC:")
print(lr_roc_auc)

print("\nConfusion Matrix:")
print(lr_cm)