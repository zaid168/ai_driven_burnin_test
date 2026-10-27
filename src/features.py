import pandas as pd


def build_features(telemetry):
    telemetry = telemetry.sort_values(
        ["chip_id", "burn_in_hour"]
    )

    # 1. Overall features: summarise the entire burn-in test
    features = telemetry.groupby("chip_id").agg(
        initial_leakage=("I_leak_uA", "first"),
        final_leakage=("I_leak_uA", "last"),
        max_leakage=("I_leak_uA", "max"),
        mean_leakage=("I_leak_uA", "mean"),

        initial_vth=("V_th_V", "first"),
        final_vth=("V_th_V", "last"),
        min_vth=("V_th_V", "min"),
        mean_vth=("V_th_V", "mean"),

        initial_delay=("t_pd_ns", "first"),
        final_delay=("t_pd_ns", "last"),
        max_delay=("t_pd_ns", "max"),
        mean_delay=("t_pd_ns", "mean"),

        initial_ddq=("I_DDQ_uA", "first"),
        final_ddq=("I_DDQ_uA", "last"),
        max_ddq=("I_DDQ_uA", "max"),
        mean_ddq=("I_DDQ_uA", "mean"),

        max_temp=("chamber_temp_C", "max"),
        mean_temp=("chamber_temp_C", "mean"),

        max_leakage_rate=("dI_dt_uA_per_h", "max"),
        max_leakage_acceleration=("d2I_dt2_uA_per_h2", "max"),
        max_vth_rate=("dVth_dt_mV_per_h", "max"),
        max_delay_rate=("dtpd_dt_ps_per_h", "max"),

        max_lot_leakage_z=("lot_z_I_leak", "max"),
        max_lot_vth_z=("lot_z_V_th", "max"),
        max_lot_delay_z=("lot_z_t_pd", "max"),
    ).reset_index()

    # 2. Overall changes
    features["leakage_change"] = (
        features["final_leakage"] - features["initial_leakage"]
    )

    features["vth_change"] = (
        features["final_vth"] - features["initial_vth"]
    )

    features["delay_change"] = (
        features["final_delay"] - features["initial_delay"]
    )

    features["ddq_change"] = (
        features["final_ddq"] - features["initial_ddq"]
    )

    # 3. Divide burn-in into four time windows
    telemetry = telemetry.copy()

    bins = [0, 48, 96, 144, 169]
    window_names = ["early", "mid1", "mid2", "late"]

    telemetry["time_window"] = pd.cut(
        telemetry["burn_in_hour"],
        bins=bins,
        labels=window_names,
        right=False
    )

    # 4. Calculate leakage behaviour in each window
    temporal = telemetry.groupby(
        ["chip_id", "time_window"],
        observed=True
    ).agg(
        window_initial_leakage=("I_leak_uA", "first"),
        window_final_leakage=("I_leak_uA", "last"),
        window_max_leakage=("I_leak_uA", "max"),
        window_mean_leakage=("I_leak_uA", "mean"),
        window_max_ddq=("I_DDQ_uA", "max"),
        window_mean_vth=("V_th_V", "mean"),
        window_max_leakage_z=("lot_z_I_leak", "max"),
    ).reset_index()

    # 5. Calculate leakage change within each window
    temporal["window_leakage_change"] = (
        temporal["window_final_leakage"]
        - temporal["window_initial_leakage"]
    )

    # 6. Convert each window into separate feature columns
    temporal_wide = temporal.pivot(
        index="chip_id",
        columns="time_window",
        values=[
            "window_leakage_change",
            "window_max_leakage",
            "window_mean_leakage",
            "window_max_ddq",
            "window_mean_vth",
            "window_max_leakage_z",
        ]
    )

    # Flatten multi-level column names
    temporal_wide.columns = [
        f"{feature}_{window}"
        for feature, window in temporal_wide.columns
    ]

    temporal_wide = temporal_wide.reset_index()

    # 7. Combine overall and temporal features
    features = features.merge(
        temporal_wide,
        on="chip_id",
        how="left"
    )

    return features


if __name__ == "__main__":
    telemetry = pd.read_csv(
        "data/zenith_burnin_telemetry.csv"
    )

    features = build_features(telemetry)

    print("Original telemetry shape:", telemetry.shape)
    print("Engineered features shape:", features.shape)
    print("Unique chips:", features["chip_id"].nunique())
    print("Missing feature values:", features.isnull().sum().sum())

    print("\nNew temporal features:")
    print([
        col for col in features.columns
        if col.endswith(("_early", "_mid1", "_mid2", "_late"))
    ])

    print("\nFirst five rows:")
    print(features.head().to_string(index=False))