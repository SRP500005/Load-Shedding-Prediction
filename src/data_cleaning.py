def clean_data(df):

    df = df.copy()

    # Remove leakage columns
    leakage_columns = [
        "load_shed_amount_mw",
        "load_shed_priority_tier"
    ]

    for column in leakage_columns:
        if column in df.columns:
            df = df.drop(columns=column)

    # Remove rows with missing lag values
    df = df.dropna(
        subset=[
            "demand_lag_24h",
            "demand_lag_168h"
        ]
    )

    # Sort data by timestamp
    df = df.sort_values("timestamp")

    # Reset index
    df = df.reset_index(drop=True)

    return df