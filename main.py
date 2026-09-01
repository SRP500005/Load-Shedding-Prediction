from src.data_loader import load_data
from src.data_validation import validate_data
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.preprocessing import prepare_data, time_based_split
from src.baseline_model import train_baseline_model
from src.logistic_regression_model import train_logistic_regression
from src.random_forest_model import train_random_forest
from src.xgboost_model import train_xgboost
from src.model_comparison import compare_models

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

# 11. Random Forest Model

(
    rf_model,
    rf_accuracy,
    rf_precision,
    rf_recall,
    rf_f1,
    rf_roc_auc,
    rf_cm
) = train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test
)


print("\n===== RANDOM FOREST =====")

print("\nAccuracy:")
print(rf_accuracy)

print("\nPrecision:")
print(rf_precision)

print("\nRecall:")
print(rf_recall)

print("\nF1 Score:")
print(rf_f1)

print("\nROC-AUC:")
print(rf_roc_auc)

print("\nConfusion Matrix:")
print(rf_cm)

# 12. XGBoost Model

(
    xgb_model,
    xgb_accuracy,
    xgb_precision,
    xgb_recall,
    xgb_f1,
    xgb_roc_auc,
    xgb_cm
) = train_xgboost(
    X_train,
    X_test,
    y_train,
    y_test
)


print("\n===== XGBOOST =====")

print("\nAccuracy:")
print(xgb_accuracy)

print("\nPrecision:")
print(xgb_precision)

print("\nRecall:")
print(xgb_recall)

print("\nF1 Score:")
print(xgb_f1)

print("\nROC-AUC:")
print(xgb_roc_auc)

print("\nConfusion Matrix:")
print(xgb_cm)

# 13. Model Comparison

baseline_metrics = [
    accuracy,
    precision,
    recall,
    f1,
    0.0
]

logistic_metrics = [
    lr_accuracy,
    lr_precision,
    lr_recall,
    lr_f1,
    lr_roc_auc
]

random_forest_metrics = [
    rf_accuracy,
    rf_precision,
    rf_recall,
    rf_f1,
    rf_roc_auc
]

xgboost_metrics = [
    xgb_accuracy,
    xgb_precision,
    xgb_recall,
    xgb_f1,
    xgb_roc_auc
]


comparison_results = compare_models(
    baseline_metrics,
    logistic_metrics,
    random_forest_metrics,
    xgboost_metrics
)