import pandas as pd

telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

telemetry = telemetry.sort_values(
    ["chip_id", "burn_in_hour"]
)

features = telemetry.groupby("chip_id").agg(
    initial_leakage=("I_leak_uA", "first"),
    final_leakage=("I_leak_uA", "last"),
    max_leakage=("I_leak_uA", "max"),
    max_leakage_z=("lot_z_I_leak", "max"),
    max_delay_z=("lot_z_t_pd", "max"),
    max_ddq=("I_DDQ_uA", "max")
).reset_index()

features["leakage_change"] = (
    features["final_leakage"]
    - features["initial_leakage"]
)

data = features.merge(
    labels[
        [
            "chip_id",
            "triage",
            "failure_mode",
            "chip_class",
            "health_score"
        ]
    ],
    on="chip_id",
    how="left"
)

print("\nTOP 15 CHIPS BY LEAKAGE CHANGE:")

print(
    data.sort_values(
        "leakage_change",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nTOP 15 CHIPS BY MAXIMUM LEAKAGE Z-SCORE:")

print(
    data.sort_values(
        "max_leakage_z",
        ascending=False
    ).head(15).to_string(index=False)
)