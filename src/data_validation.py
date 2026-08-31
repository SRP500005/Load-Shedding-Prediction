import pandas as pd


def validate_data(df):

    print("\n===== DATA VALIDATION =====")

    print("\nRows:")
    print(df.shape[0])

    print("\nColumns:")
    print(df.shape[1])

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nTarget Distribution:")
    print(df["load_shed_flag"].value_counts())

    print("\nTarget Percentage:")
    print(
        df["load_shed_flag"].value_counts(normalize=True) * 100
    )

    print("\nTimestamp Range:")
    print("Start:", df["timestamp"].min())
    print("End:", df["timestamp"].max())