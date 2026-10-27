import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

from features import build_features

# Load data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

# Build shared features
features = build_features(telemetry)

# Attach target labels
data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id",
    how="left",
    validate="one_to_one"
)

# Separate inputs and target
X = data.drop(columns=["chip_id", "triage"])
y = data["triage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    class_weight={
        "GREEN": 1.0,
        "AMBER": 3.0,
        "RED": 2.0
    }
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, predictions))

print("CONFUSION MATRIX:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["GREEN", "AMBER", "RED"]
))

print(
    "BALANCED ACCURACY:",
    balanced_accuracy_score(y_test, predictions)
)

# Save model
joblib.dump(model, "zenith_model.pkl")
print("\nModel saved to zenith_model.pkl")

# Feature importance analysis
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES:")
print(importance.head(15).to_string(index=False))

importance.to_csv("feature_importance.csv", index=False)
print("\nFeature importance saved to feature_importance.csv")