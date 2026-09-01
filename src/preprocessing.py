import pandas as pd


def prepare_data(df):

    df = df.copy()

    # Target column
    target = "load_shed_flag"

    # Timestamp model me direct use nahi karenge
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Categorical columns
    categorical_columns = [
        "season",
        "feeder_id"
    ]

    # One-hot encoding
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    # Features and target separate karo
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def time_based_split(X, y, train_ratio=0.8):

    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test