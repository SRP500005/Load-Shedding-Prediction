import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)


def optimize_random_forest_threshold(
    X_train,
    X_test,
    y_train,
    y_test
):

    # 1. Chronological train-validation split
    split_index = int(len(X_train) * 0.8)

    X_subtrain = X_train.iloc[:split_index]
    X_val = X_train.iloc[split_index:]

    y_subtrain = y_train.iloc[:split_index]
    y_val = y_train.iloc[split_index:]

    print("\n===== THRESHOLD OPTIMIZATION =====")

    print("\nSub-training shape:")
    print(X_subtrain.shape)

    print("\nValidation shape:")
    print(X_val.shape)

    print("\nSub-training target distribution:")
    print(y_subtrain.value_counts())

    print("\nValidation target distribution:")
    print(y_val.value_counts())

    # 2. Train RF using only sub-training data
    validation_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    validation_model.fit(
        X_subtrain,
        y_subtrain
    )

    # 3. Validation probabilities
    val_probabilities = validation_model.predict_proba(
        X_val
    )[:, 1]

    # 4. Calculate Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(
        y_val,
        val_probabilities
    )

    # PR-AUC / Average Precision
    pr_auc = average_precision_score(
        y_val,
        val_probabilities
    )

    # 5. Calculate F1 for every possible threshold
    f1_scores = (
        2 * precisions[:-1] * recalls[:-1]
    ) / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )

    best_index = np.argmax(f1_scores)

    best_threshold = thresholds[best_index]

    best_validation_f1 = f1_scores[best_index]

    print("\nValidation PR-AUC:")
    print(pr_auc)

    print("\nBest Threshold:")
    print(best_threshold)

    print("\nBest Validation F1:")
    print(best_validation_f1)

    # 6. Save Precision-Recall curve
    os.makedirs(
        "reports",
        exist_ok=True
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recalls,
        precisions
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Random Forest Precision-Recall Curve"
    )

    plt.tight_layout()

    plt.savefig(
        "reports/random_forest_precision_recall_curve.png"
    )

    plt.close()

    # 7. Train final RF using ALL training data
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(
        X_train,
        y_train
    )

    # 8. Get probabilities on untouched test data
    test_probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    # 9. Apply validation-selected threshold
    optimized_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # 10. Final test metrics
    accuracy = accuracy_score(
        y_test,
        optimized_predictions
    )

    final_precision = precision_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    final_recall = recall_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    final_f1 = f1_score(
        y_test,
        optimized_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        optimized_predictions
    )

    print(
        "\n===== RANDOM FOREST - OPTIMIZED THRESHOLD ====="
    )

    print("\nThreshold:")
    print(best_threshold)

    print("\nAccuracy:")
    print(accuracy)

    print("\nPrecision:")
    print(final_precision)

    print("\nRecall:")
    print(final_recall)

    print("\nF1 Score:")
    print(final_f1)

    print("\nConfusion Matrix:")
    print(cm)

    print(
        "\nSaved: reports/random_forest_precision_recall_curve.png"
    )

    return (
        final_model,
        best_threshold,
        pr_auc,
        accuracy,
        final_precision,
        final_recall,
        final_f1,
        cm
    )