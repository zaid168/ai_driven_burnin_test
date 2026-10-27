import pandas as pd

from features import build_features


# 1. Load telemetry, labels, and misclassified chip IDs
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")
errors = pd.read_csv("misclassified_chips.csv")

# 2. Generate features for all chips
features = build_features(telemetry)

# 3. Select the chips that were misclassified
error_chips = errors[["chip_id", "actual", "predicted"]].drop_duplicates(
    subset="chip_id"
)

investigation = error_chips.merge(
    features,
    on="chip_id"
)

# 4. Select useful electrical and temporal features
selected_columns = [
    "chip_id",
    "actual",
    "predicted",
    "max_leakage",
    "leakage_change",
    "max_ddq",
    "ddq_change",
    "delay_change",
    "vth_change",
    "max_leakage_rate",
    "window_leakage_change_early",
    "window_leakage_change_mid1",
    "window_leakage_change_mid2",
    "window_leakage_change_late",
    "window_max_leakage_z_late",
]

print("\nMISCLASSIFIED CHIP CHARACTERISTICS")

print(
    investigation[selected_columns]
    .round(4)
    .to_string(index=False)
)

# 5. Compare misclassified chips with all chips in their actual class
comparison = investigation.merge(
    labels[["chip_id", "triage"]],
    on="chip_id",
    how="left"
)

print("\nERROR COUNTS BY ACTUAL CLASS")
print(comparison["actual"].value_counts())

# Save the detailed results
investigation[selected_columns].to_csv(
    "misclassified_chip_features.csv",
    index=False
)

print("\nSaved misclassified_chip_features.csv")