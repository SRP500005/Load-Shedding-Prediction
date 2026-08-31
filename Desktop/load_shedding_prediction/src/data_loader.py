import pandas as pd


def load_data(file_path):

    df = pd.read_csv(file_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df