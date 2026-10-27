
import json
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from features import build_features


# 1. Load data
ROOT = Path(__file__).resolve().parent.parent

telemetry = pd.read_csv(
    ROOT / "data" / "zenith_burnin_telemetry.csv"
)
labels = pd.read_csv(
    ROOT / "data" / "zenith_chip_labels.csv"
)

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

X = data.drop(columns=["chip_id", "triage"])
y = data["triage"]

classes = ["GREEN", "AMBER", "RED"]


# 2. Configure cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 3. Train and evaluate each fold
fold_results = []
confusion_matrices = []

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
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )
    macro_f1 = f1_score(
        y_test, predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    fold_results.append({
        "fold": fold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1
    })

    cm = confusion_matrix(
        y_test, predictions, labels=classes
    )

    confusion_matrices.append({
        "fold": fold,
        "matrix": cm.tolist()
    })

    print(f"\n{'=' * 15} FOLD {fold} {'=' * 15}")
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Balanced accuracy: {balanced_accuracy:.3f}")
    print(f"Macro F1:          {macro_f1:.3f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=classes,
        zero_division=0
    ))

    print("Confusion matrix (GREEN, AMBER, RED):")
    print(cm)


# 4. Save fold results
results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    ROOT / "cross_validation_results.csv",
    index=False
)

with open(
    ROOT / "cross_validation_confusion_matrices.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(confusion_matrices, file, indent=2)


# 5. Print final summary
print("\n========== CROSS-VALIDATION SUMMARY ==========")

for metric in [
    "accuracy",
    "balanced_accuracy",
    "macro_f1"
]:
    mean_score = results_df[metric].mean()
    std_score = results_df[metric].std(ddof=0)

    print(
        f"{metric}: "
        f"mean={mean_score:.3f}, "
        f"std={std_score:.3f}"
    )

print("\nSaved:")
print("- cross_validation_results.csv")
print("- cross_validation_confusion_matrices.json")
