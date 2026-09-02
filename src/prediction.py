import joblib
import pandas as pd


MODEL_PATH = "models/final_random_forest.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
THRESHOLD_PATH = "models/classification_threshold.pkl"


def load_prediction_artifacts():

    model = joblib.load(MODEL_PATH)

    feature_columns = joblib.load(
        FEATURES_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    return (
        model,
        feature_columns,
        threshold
    )


def prepare_input_data(
    input_data,
    feature_columns
):

    input_df = pd.DataFrame(
        [input_data]
    )

    # -----------------------------
    # Season one-hot encoding
    # -----------------------------

    season = input_data.get(
        "season",
        "Autumn"
    )

    input_df = input_df.drop(
        columns=["season"],
        errors="ignore"
    )

    for season_name in [
        "Spring",
        "Summer",
        "Winter"
    ]:

        column_name = (
            f"season_{season_name}"
        )

        input_df[column_name] = int(
            season == season_name
        )

    # Autumn is the dropped baseline category

    # -----------------------------
    # Feeder one-hot encoding
    # -----------------------------

    feeder = input_data.get(
        "feeder_id",
        "FDR_01"
    )

    input_df = input_df.drop(
        columns=["feeder_id"],
        errors="ignore"
    )

    for feeder_number in range(
        2,
        11
    ):

        feeder_name = (
            f"FDR_{feeder_number:02d}"
        )

        column_name = (
            f"feeder_id_{feeder_name}"
        )

        input_df[column_name] = int(
            feeder == feeder_name
        )

    # FDR_01 is baseline category

    # -----------------------------
    # Match training columns
    # -----------------------------

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return input_df


def predict_load_shedding(
    input_data
):

    (
        model,
        feature_columns,
        threshold
    ) = load_prediction_artifacts()

    prepared_data = prepare_input_data(
        input_data,
        feature_columns
    )

    probability = model.predict_proba(
        prepared_data
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold
    }