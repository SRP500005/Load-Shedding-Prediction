import os
import matplotlib.pyplot as plt


def perform_eda(df):

    print("\n===== EXPLORATORY DATA ANALYSIS =====")

    # Reports folder create karo agar exist nahi karta
    os.makedirs("reports", exist_ok=True)

    # -----------------------------------
    # 1. Target Distribution
    # -----------------------------------

    target_counts = df["load_shed_flag"].value_counts()

    print("\nLoad Shedding Distribution:")
    print(target_counts)

    plt.figure(figsize=(6, 4))

    target_counts.plot(kind="bar")

    plt.title("Load Shedding Distribution")
    plt.xlabel("Load Shed Flag")
    plt.ylabel("Number of Observations")

    plt.tight_layout()

    plt.savefig(
        "reports/target_distribution.png"
    )

    plt.close()

    print(
        "Saved: reports/target_distribution.png"
    )


    # -----------------------------------
    # 2. Total Demand Distribution
    # -----------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["total_demand_mw"],
        bins=50
    )

    plt.title("Total Electricity Demand Distribution")
    plt.xlabel("Total Demand (MW)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "reports/demand_distribution.png"
    )

    plt.close()

    print(
        "Saved: reports/demand_distribution.png"
    )


    # -----------------------------------
    # 3. Reserve Margin vs Load Shedding
    # -----------------------------------

    reserve_analysis = (
        df.groupby("load_shed_flag")["reserve_margin_pct"]
        .mean()
    )

    print("\nAverage Reserve Margin:")
    print(reserve_analysis)

    reserve_analysis.plot(
        kind="bar",
        figsize=(6, 4)
    )

    plt.title(
        "Average Reserve Margin vs Load Shedding"
    )

    plt.xlabel("Load Shed Flag")
    plt.ylabel("Average Reserve Margin (%)")

    plt.tight_layout()

    plt.savefig(
        "reports/reserve_margin_vs_load_shedding.png"
    )

    plt.close()

    print(
        "Saved: reports/reserve_margin_vs_load_shedding.png"
    )


    # -----------------------------------
    # 4. Demand vs Load Shedding
    # -----------------------------------

    demand_analysis = (
        df.groupby("load_shed_flag")["total_demand_mw"]
        .mean()
    )

    print("\nAverage Electricity Demand:")
    print(demand_analysis)

    demand_analysis.plot(
        kind="bar",
        figsize=(6, 4)
    )

    plt.title(
        "Average Demand vs Load Shedding"
    )

    plt.xlabel("Load Shed Flag")
    plt.ylabel("Average Demand (MW)")

    plt.tight_layout()

    plt.savefig(
        "reports/demand_vs_load_shedding.png"
    )

    plt.close()

    print(
        "Saved: reports/demand_vs_load_shedding.png"
    )