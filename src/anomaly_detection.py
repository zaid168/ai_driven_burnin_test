import pandas as pd
from sklearn.ensemble import IsolationForest
from features import build_features

# 1. Load telemetry
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")

# 2. Build chip-level features
features = build_features(telemetry)

# 3. Prepare model inputs
X = features.drop(columns=["chip_id"])

# 4. Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.10,
    random_state=42
)

model.fit(X)
import joblib

joblib.dump(model, "zenith_anomaly_model.pkl")

print("Anomaly model saved successfully!")

# 5. Generate predictions and scores
features["anomaly_prediction"] = model.predict(X)
features["anomaly_score"] = model.decision_function(X)

features["anomaly_status"] = features["anomaly_prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

# 6. Compare against existing labels for analysis
labels = pd.read_csv("data/zenith_chip_labels.csv")

results = features.merge(
    labels[["chip_id", "triage", "failure_mode"]],
    on="chip_id",
    how="left",
    validate="one_to_one"
)

# 7. Display results
print("TOTAL CHIPS:", len(results))

print("\nANOMALY COUNTS:")
print(results["anomaly_status"].value_counts())

print("\nANOMALIES BY TRIAGE:")
print(pd.crosstab(
    results["anomaly_status"],
    results["triage"]
))

print("\nTOP 15 MOST ANOMALOUS CHIPS:")
print(
    results.sort_values("anomaly_score")[
        [
            "chip_id",
            "max_leakage",
            "leakage_change",
            "max_ddq",
            "delay_change",
            "anomaly_score",
            "anomaly_status",
        ]
    ].head(15).to_string(index=False)
)

