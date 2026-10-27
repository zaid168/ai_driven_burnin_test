import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

from features import build_features

# 1. Load the data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

# 2. Build one feature row per chip
features = build_features(telemetry)
data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

# 3. Select the two classes we want to compare
comparison = data[data["triage"].isin(["GREEN", "AMBER"])]

# 4. Features to visualize
selected_features = [
    "delay_change",
    "max_ddq",
    "window_leakage_change_mid2",
    "window_leakage_change_late",
]

for feature in selected_features:

    # Separate numerical summaries for both classes
    print(f"\n{'=' * 65}")
    print(f"FEATURE: {feature}")
    print("=" * 65)

    summary = comparison.groupby("triage")[feature].agg(
        count="count",
        minimum="min",
        q1=lambda x: x.quantile(0.25),
        median="median",
        q3=lambda x: x.quantile(0.75),
        maximum="max",
        mean="mean"
    )

    print(summary.round(4).to_string())

    # Draw the graph
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=comparison,
        x="triage",
        y=feature,
        order=["GREEN", "AMBER"],
        width=0.4,
        showfliers=False,
        ax=ax
    )

    # Show individual chips as dots
    sns.stripplot(
        data=comparison,
        x="triage",
        y=feature,
        order=["GREEN", "AMBER"],
        jitter=0.12,
        size=5,
        alpha=0.75,
        ax=ax
    )

    # Number the vertical axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))

    # Add class counts to the labels
    green_count = (comparison["triage"] == "GREEN").sum()
    amber_count = (comparison["triage"] == "AMBER").sum()

    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"GREEN (n={green_count})",
        f"AMBER (n={amber_count})"
    ])

    ax.set_title(f"{feature}: GREEN vs AMBER")
    ax.set_xlabel("Actual chip class")
    ax.set_ylabel(feature)

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save and display
    filename = f"{feature}_precise.png"
    plt.savefig(filename, dpi=200)
    plt.show()
    plt.close()

    print(f"\nGraph saved as {filename}")

print("\nAll comparisons completed.")