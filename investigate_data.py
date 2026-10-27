import pandas as pd

# Load the existing data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

# Sort measurements chronologically for each chip
telemetry = telemetry.sort_values(
    ["chip_id", "burn_in_hour"]
)

# Aggregate telemetry into one row per chip
features = telemetry.groupby("chip_id").agg(
    initial_leakage=("I_leak_uA", "first"),
    final_leakage=("I_leak_uA", "last"),
    max_leakage=("I_leak_uA", "max"),
    mean_leakage=("I_leak_uA", "mean"),

    initial_vth=("V_th_V", "first"),
    final_vth=("V_th_V", "last"),

    initial_delay=("t_pd_ns", "first"),
    final_delay=("t_pd_ns", "last"),
    max_delay=("t_pd_ns", "max"),

    initial_ddq=("I_DDQ_uA", "first"),
    final_ddq=("I_DDQ_uA", "last"),
    max_ddq=("I_DDQ_uA", "max"),

    max_leakage_rate=("dI_dt_uA_per_h", "max"),
    max_leakage_z=("lot_z_I_leak", "max"),
    max_delay_z=("lot_z_t_pd", "max")
).reset_index()

# Calculate changes during burn-in
features["leakage_change"] = (
    features["final_leakage"]
    - features["initial_leakage"]
)

features["delay_change"] = (
    features["final_delay"]
    - features["initial_delay"]
)

features["ddq_change"] = (
    features["final_ddq"]
    - features["initial_ddq"]
)

# Attach the target labels
data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id",
    how="left",
    validate="one_to_one"
)

# Compare measurements across triage groups
columns = [
    "initial_leakage",
    "final_leakage",
    "max_leakage",
    "leakage_change",
    "final_vth",
    "delay_change",
    "max_delay",
    "ddq_change",
    "max_leakage_rate",
    "max_leakage_z",
    "max_delay_z"
]

print("\nMEAN FEATURE VALUES BY TRIAGE:")
print(
    data.groupby("triage")[columns]
    .mean()
    .round(3)
    .to_string()
)

print("\nMEDIAN FEATURE VALUES BY TRIAGE:")
print(
    data.groupby("triage")[columns]
    .median()
    .round(3)
    .to_string()
)