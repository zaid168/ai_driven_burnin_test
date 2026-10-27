import pandas as pd

from features import build_features


# 1. Load data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

# 2. Select the features we want to investigate
selected_features = [
    "delay_change",
    "max_lot_delay_z",
    "final_delay",
    "max_ddq",
    "window_max_ddq_mid1",
    "window_leakage_change_mid2",
    "window_leakage_change_late",
]

# 3. Compare AMBER and GREEN
comparison = data[data["triage"].isin(["AMBER", "GREEN"])]

summary = comparison.groupby("triage")[selected_features].agg(
    ["mean", "median", "min", "max"]
)

print("\nAMBER VS GREEN FEATURE COMPARISON")
print(summary.round(4).to_string())

# 4. Save the results
summary.to_csv("amber_green_comparison.csv")

print("\nSaved amber_green_comparison.csv")