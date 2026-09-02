import joblib
import pandas as pd


def load_prediction_artifacts():

    model = joblib.load(
        "models/final_random_forest.pkl"
    )

    feature_columns = joblib.load(
        "models/feature_columns.pkl"
    )

    threshold = joblib.load(
        "models/classification_threshold.pkl"
    )

    return (
        model,
        feature_columns,
        threshold
    )


def predict_load_shedding(input_data):

    (
        model,
        feature_columns,
        threshold
    ) = load_prediction_artifacts()

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )

    # Make sure columns match training features
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict probability
    probability = model.predict_proba(
        input_df
    )[0][1]

    # Apply threshold
    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold
    }