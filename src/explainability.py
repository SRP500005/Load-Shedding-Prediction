import os
import pandas as pd
import matplotlib.pyplot as plt
import shap


def explain_random_forest(
    model,
    X_train,
    X_test
):

    print("\n===== MODEL EXPLAINABILITY =====")

    os.makedirs(
        "reports",
        exist_ok=True
    )

    # ==========================================
    # 1. RANDOM FOREST FEATURE IMPORTANCE
    # ==========================================

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 Important Features:")

    print(
        feature_importance.head(15).to_string(
            index=False
        )
    )

    # Save feature importance CSV
    feature_importance.to_csv(
        "reports/random_forest_feature_importance.csv",
        index=False
    )

    # Plot top 15 features
    top_features = feature_importance.head(15)

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.title(
        "Random Forest - Top 15 Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        "reports/random_forest_feature_importance.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSaved: reports/random_forest_feature_importance.csv"
    )

    print(
        "Saved: reports/random_forest_feature_importance.png"
    )

    # ==========================================
    # 2. SHAP EXPLAINABILITY
    # ==========================================

    print("\nCalculating SHAP values...")

    # Use a sample so SHAP does not become too slow
    sample_size = min(
        1000,
        len(X_test)
    )

    X_shap = X_test.sample(
        n=sample_size,
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_shap
    )

    # For binary classification,
    # select SHAP values for positive class
    if len(shap_values.values.shape) == 3:
        positive_shap_values = shap_values.values[:, :, 1]

        shap_explanation = shap.Explanation(
            values=positive_shap_values,
            base_values=shap_values.base_values[:, 1],
            data=X_shap.values,
            feature_names=X_shap.columns
        )

    else:
        shap_explanation = shap_values

    # ==========================================
    # 3. SHAP BAR PLOT
    # ==========================================

    shap.plots.bar(
        shap_explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "reports/shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved: reports/shap_feature_importance.png"
    )

    # ==========================================
    # 4. SHAP BEESWARM PLOT
    # ==========================================

    shap.plots.beeswarm(
        shap_explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "reports/shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved: reports/shap_summary.png"
    )

    print(
        "\nSHAP analysis completed successfully."
    )

    return feature_importance