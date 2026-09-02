import os
import joblib


def save_final_model(
    model,
    feature_columns,
    threshold=0.50
):

    os.makedirs(
        "models",
        exist_ok=True
    )

    model_path = "models/final_random_forest.pkl"
    features_path = "models/feature_columns.pkl"
    threshold_path = "models/classification_threshold.pkl"

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        list(feature_columns),
        features_path
    )

    joblib.dump(
        threshold,
        threshold_path
    )

    print("\n===== FINAL MODEL SAVED =====")

    print("\nModel:")
    print(model_path)

    print("\nFeature Columns:")
    print(features_path)

    print("\nClassification Threshold:")
    print(threshold_path)

    print("\nThreshold Value:")
    print(threshold)

    return (
        model_path,
        features_path,
        threshold_path
    )