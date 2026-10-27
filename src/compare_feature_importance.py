import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from features import build_features


# 1. Load and prepare data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

X = data.drop(columns=["chip_id", "triage"])
y = data["triage"]


# 2. Prepare five-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

importance_tables = []


# 3. Train a model for each fold
for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y),
    start=1
):
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })

    fold_importance["fold"] = fold

    importance_tables.append(fold_importance)

    print(f"\nTOP 10 FEATURES — FOLD {fold}")

    print(
        fold_importance
        .sort_values("importance", ascending=False)
        .head(10)
        .to_string(index=False)
    )


# 4. Compare importance across folds
all_importance = pd.concat(
    importance_tables,
    ignore_index=True
)

summary = (
    all_importance
    .groupby("feature")["importance"]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)

print("\nTOP 15 FEATURES ACROSS ALL FOLDS")
print(summary.head(15).round(4).to_string())

summary.to_csv("feature_importance_stability.csv")

print("\nSaved feature_importance_stability.csv")