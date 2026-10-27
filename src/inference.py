from pathlib import Path

import joblib
import pandas as pd

from features import build_features


# 1. Define paths relative to the project root
ROOT = Path(__file__).resolve().parent.parent

TELEMETRY_PATH = ROOT / "data" / "zenith_burnin_telemetry.csv"
RISK_MODEL_PATH = ROOT / "zenith_model.pkl"
ANOMALY_MODEL_PATH = ROOT / "zenith_anomaly_model.pkl"


def generate_predictions():
    """Generate chip-level risk predictions and anomaly results."""

    # 2. Load the saved models
    risk_model = joblib.load(RISK_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)


    # Load telemetry
    telemetry = pd.read_csv(TELEMETRY_PATH)

    # Validate the input
    if telemetry.empty:
        raise ValueError("Telemetry file is empty.")

    required_columns = [
        "chip_id",
        "burn_in_hour",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in telemetry.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Telemetry is missing required columns: {missing_columns}"
        )

    if telemetry["chip_id"].isna().any():
        raise ValueError("Telemetry contains missing chip IDs.")

    if telemetry["burn_in_hour"].isna().any():
        raise ValueError("Telemetry contains missing burn-in hours.")

    # Generate chip-level features
    features = build_features(telemetry)
    

    # 4. Validate the generated features
    if features.empty:
        raise ValueError("Feature engineering produced no chip records.")

    if features["chip_id"].isna().any():
        raise ValueError("Some chip records have missing chip_id values.")

    if features["chip_id"].duplicated().any():
        raise ValueError("Expected one feature row per chip, but duplicates exist.")

    # 5. Prepare the correct input features for each model
    risk_columns = list(risk_model.feature_names_in_)
    anomaly_columns = list(anomaly_model.feature_names_in_)

    missing_risk = set(risk_columns) - set(features.columns)
    missing_anomaly = set(anomaly_columns) - set(features.columns)

    if missing_risk:
        raise ValueError(
            f"Missing features required by the risk model: {sorted(missing_risk)}"
        )

    if missing_anomaly:
        raise ValueError(
            f"Missing features required by the anomaly model: {sorted(missing_anomaly)}"
        )

    X_risk = features[risk_columns]
    X_anomaly = features[anomaly_columns]

    # 6. Generate predictions
    results = features[["chip_id"]].copy()

    results["predicted_triage"] = risk_model.predict(X_risk)

    results["anomaly_prediction"] = anomaly_model.predict(X_anomaly)
    results["anomaly_score"] = anomaly_model.decision_function(X_anomaly)

    results["anomaly_status"] = results["anomaly_prediction"].map({
        1: "NORMAL",
        -1: "ANOMALY",
    })

    # 7. Include important engineering indicators
    indicator_columns = [
        "max_leakage",
        "leakage_change",
        "max_ddq",
        "delay_change",
    ]

    for column in indicator_columns:
        if column in features.columns:
            results[column] = features[column]

    # 8. Assign a recommendation
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
        axis=1,
    )

    # 9. Validate the final output
    if results.isnull().any().any():
        raise ValueError("Prediction results contain missing values.")

    if len(results) != features["chip_id"].nunique():
        raise ValueError("Prediction count does not match the number of chips.")

    return results


# Run this file directly to test the inference pipeline
if __name__ == "__main__":
    results = generate_predictions()

    print("\nCOMBINED PREDICTIONS:")
    print(results.to_string(index=False))

    print("\nRECOMMENDATION COUNTS:")
    print(results["recommendation"].value_counts())

    output_path = ROOT / "chip_predictions.csv"
    results.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")