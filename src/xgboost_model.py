from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def train_xgboost(X_train, X_test, y_train, y_test):

    # Class imbalance ratio
    negative_class = (y_train == 0).sum()
    positive_class = (y_train == 1).sum()

    scale_pos_weight = negative_class / positive_class

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    predictions = model.predict(
        X_test
    )

    # Probabilities
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Evaluation
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return (
        model,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        cm
    )