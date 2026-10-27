import pandas as pd

labels = pd.read_csv("data/zenith_chip_labels.csv")

print("\nFAILURE MODE vs TRIAGE:")
print(pd.crosstab(labels["failure_mode"], labels["triage"]))

print("\nFAILURE MODE vs CHIP CLASS:")
print(pd.crosstab(labels["failure_mode"], labels["chip_class"]))

print("\nFAILURE MODE vs LATENT DEFECT FLAG:")
print(pd.crosstab(labels["failure_mode"], labels["latent_defect_flag"]))

print("\nHEALTH SCORE BY FAILURE MODE:")
print(
    labels.groupby("failure_mode")["health_score"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\nFAILURE PROBABILITY BY FAILURE MODE:")
print(
    labels.groupby("failure_mode")["failure_probability_pct"]
    .agg(["mean", "median", "min", "max"])
    .round(2)
)