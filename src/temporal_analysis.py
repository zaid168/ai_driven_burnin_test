import pandas as pd

# 1. Load telemetry
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")

# 2. Define burn-in windows
bins = [0, 48, 96, 144, 169]

labels = [
    "0-47h",
    "48-95h",
    "96-143h",
    "144-168h"
]

telemetry["time_window"] = pd.cut(
    telemetry["burn_in_hour"],
    bins=bins,
    labels=labels,
    right=False
)

# 3. Sort measurements chronologically
telemetry = telemetry.sort_values(
    ["chip_id", "burn_in_hour"]
)

# 4. Aggregate measurements per chip per window
temporal_features = (
    telemetry.groupby(
        ["chip_id", "time_window"],
        observed=True
    )
    .agg(
        initial_leakage=("I_leak_uA", "first"),
        final_leakage=("I_leak_uA", "last"),
        max_leakage=("I_leak_uA", "max"),
        mean_leakage=("I_leak_uA", "mean"),
        max_ddq=("I_DDQ_uA", "max"),
        mean_vth=("V_th_V", "mean"),
        max_leakage_z=("lot_z_I_leak", "max"),
        measurements=("record_id", "count")
    )
    .reset_index()
)

# 5. Calculate leakage change within each window
temporal_features["leakage_change"] = (
    temporal_features["final_leakage"]
    - temporal_features["initial_leakage"]
)

# 6. Display the window summary
print("\nMEASUREMENTS PER WINDOW:")
print(
    temporal_features.groupby(
        "time_window",
        observed=True
    )["measurements"].agg(["count", "min", "max"])
)

# 7. Inspect one chip across all windows
chip_id = "ZN-03-019"

print(f"\nTEMPORAL BEHAVIOUR OF {chip_id}:")

chip_history = temporal_features[
    temporal_features["chip_id"] == chip_id
]

print(chip_history.to_string(index=False))

# 8. Save results
temporal_features.to_csv(
    "temporal_chip_analysis.csv",
    index=False
)

print("\nSaved temporal_chip_analysis.csv")