import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from features import build_features
from pathlib import Path


# 1. Load data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id",
)

X = data.drop(columns=["chip_id", "triage"])
y = data["triage"]

classes = ["GREEN", "AMBER", "RED"]


# 2. Configure cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# 3. Compare class-weight strategies
strategies = {
    "Balanced": "balanced",
    "No class weighting": None,
    "Custom weighting": {
        "GREEN": 1.0,
        "AMBER": 3.0,
        "RED": 2.0,
    },
}

prediction_records = []
results = []


for strategy_name, class_weight in strategies.items():

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight=class_weight,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        for position, actual, predicted in zip(
            test_idx,
            y_test,
            predictions
        ):
            prediction_records.append({
                "strategy": strategy_name,
                "fold": fold,
                "chip_id": data.iloc[position]["chip_id"],
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
            })

        results.append({
            "strategy": strategy_name,
            "fold": fold,
            "accuracy": accuracy_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            ),
            "amber_precision": precision_score(
                y_test,
                predictions,
                labels=["AMBER"],
                average="micro",
                zero_division=0,
            ),
            "amber_recall": recall_score(
                y_test,
                predictions,
                labels=["AMBER"],
                average="micro",
                zero_division=0,
            ),
            "red_recall": recall_score(
                y_test,
                predictions,
                labels=["RED"],
                average="micro",
                zero_division=0,
            ),
        })

        print(
            f"{strategy_name} | Fold {fold} | "
            f"Accuracy: {accuracy_score(y_test, predictions):.3f}"
        )


# 4. Save individual fold results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "class_weight_comparison.csv",
    index=False,
)


# 5. Summarize mean scores
metrics = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "amber_precision",
    "amber_recall",
    "red_recall",
]

summary = results_df.groupby("strategy")[metrics].mean()

print("\n========== MEAN FIVE-FOLD RESULTS ==========")
print(summary.round(3).to_string())

print("\n========== AMBER RECALL BY FOLD ==========")
print(
    results_df.pivot(
        index="fold",
        columns="strategy",
        values="amber_recall",
    ).round(3).to_string()
)

print("\nSaved class_weight_comparison.csv")

# Save predictions from every validation fold
predictions_df = pd.DataFrame(prediction_records)

predictions_df.to_csv(
    "class_weight_predictions.csv",
    index=False,
)

# Save only incorrect predictions for easier investigation
errors_df = predictions_df[
    predictions_df["correct"] == False
].copy()

errors_df.to_csv(
    "class_weight_errors.csv",
    index=False,
)

print("\nSaved class_weight_predictions.csv")
print("Saved class_weight_errors.csv")