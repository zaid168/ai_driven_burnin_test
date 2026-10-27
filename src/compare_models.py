import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

from features import build_features


# 1. Load data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

y = data["triage"]

# 2. Model A: original features only
original_features = [
    col for col in features.columns
    if not col.startswith("window_")
]

X_original = data[original_features].drop(columns=["chip_id"])

# 3. Model B: original + temporal features
X_temporal = data.drop(
    columns=["chip_id", "triage"]
)

# 4. Use the same split for both models
X_train_a, X_test_a, y_train, y_test = train_test_split(
    X_original,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train_b, X_test_b, _, _ = train_test_split(
    X_temporal,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Train both models
model_a = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model_b = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

model_a.fit(X_train_a, y_train)
model_b.fit(X_train_b, y_train)

# 6. Evaluate
pred_a = model_a.predict(X_test_a)
pred_b = model_b.predict(X_test_b)

score_a = balanced_accuracy_score(y_test, pred_a)
score_b = balanced_accuracy_score(y_test, pred_b)

print("\nMODEL A: ORIGINAL FEATURES")
print("Number of features:", X_original.shape[1])
print("Balanced accuracy:", score_a)
print(classification_report(y_test, pred_a, zero_division=0))

print("\nMODEL B: ORIGINAL + TEMPORAL FEATURES")
print("Number of features:", X_temporal.shape[1])
print("Balanced accuracy:", score_b)
print(classification_report(y_test, pred_b, zero_division=0))

print("\nCOMPARISON")
print(f"Original model: {score_a:.3f}")
print(f"Temporal model: {score_b:.3f}")
print(f"Difference: {score_b - score_a:+.3f}")