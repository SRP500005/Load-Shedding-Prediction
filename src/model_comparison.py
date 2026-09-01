import os
import pandas as pd
import matplotlib.pyplot as plt


def compare_models(
    baseline_metrics,
    logistic_metrics,
    random_forest_metrics,
    xgboost_metrics
):

    print("\n===== MODEL COMPARISON =====")

    # Create comparison table
    results = pd.DataFrame({
        "Baseline": baseline_metrics,
        "Logistic Regression": logistic_metrics,
        "Random Forest": random_forest_metrics,
        "XGBoost": xgboost_metrics
    })

    # Rows = evaluation metrics
    results.index = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    # Transpose for easier reading
    results = results.T

    print("\nModel Performance:")
    print(results.round(4))

    # Create reports folder if needed
    os.makedirs("reports", exist_ok=True)

    # Save results as CSV
    results.to_csv(
        "reports/model_comparison.csv"
    )

    # Create comparison graph
    results.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.title("Machine Learning Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "reports/model_comparison.png"
    )

    plt.close()

    print(
        "\nSaved: reports/model_comparison.csv"
    )

    print(
        "Saved: reports/model_comparison.png"
    )

    # Find best model based on F1 score
    best_model = results["F1 Score"].idxmax()

    best_f1 = results.loc[
        best_model,
        "F1 Score"
    ]

    print("\nBest Model based on F1 Score:")
    print(best_model)

    print("\nBest F1 Score:")
    print(round(best_f1, 4))

    return results