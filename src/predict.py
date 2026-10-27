import joblib
import pandas as pd

from features import build_features

# 1. Load the trained models
risk_model = joblib.load("zenith_model.pkl")
anomaly_model = joblib.load("zenith_anomaly_model.pkl")

# 2. Load telemetry
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")

# 3. Generate chip-level features
features = build_features(telemetry)

# 4. Prepare separate inputs for each model
risk_columns = list(risk_model.feature_names_in_)
X_risk = features[risk_columns]

anomaly_columns = list(anomaly_model.feature_names_in_)
X_anomaly = features[anomaly_columns]

# 5. Predict risk category using Random Forest
features["predicted_triage"] = risk_model.predict(X_risk)

# 6. Detect anomalies using Isolation Forest
features["anomaly_prediction"] = anomaly_model.predict(X_anomaly)
features["anomaly_score"] = anomaly_model.decision_function(X_anomaly)

features["anomaly_status"] = features["anomaly_prediction"].map({
    1: "NORMAL",
    -1: "ANOMALY"
})

# 7. Combine the outputs
results = features[
    [
        "chip_id",
        "predicted_triage",
        "anomaly_status",
        "anomaly_score",
        "max_leakage",
        "leakage_change",
        "max_ddq",
        "delay_change",
    ]
].copy()

# 8. Generate an initial review recommendation
def make_recommendation(row):
    if row["predicted_triage"] == "RED":
        return "REJECT_CANDIDATE"

    if (
        row["predicted_triage"] == "AMBER"
        or row["anomaly_status"] == "ANOMALY"
    ):
        return "REVIEW"

    return "PASS_CANDIDATE"


results["recommendation"] = results.apply(
    make_recommendation,
    axis=1
)

# 9. Display results
print("\nCOMBINED PREDICTIONS:")
print(
    results.sort_values(
        ["predicted_triage", "anomaly_score"]
    ).to_string(index=False)
)

print("\nRECOMMENDATION COUNTS:")
print(results["recommendation"].value_counts())

# 10. Save results
results.to_csv("chip_predictions.csv", index=False)

print("\nResults saved to chip_predictions.csv")
