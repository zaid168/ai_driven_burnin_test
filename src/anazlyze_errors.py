import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from features import build_features


# 1. Load the data
telemetry = pd.read_csv("data/zenith_burnin_telemetry.csv")
labels = pd.read_csv("data/zenith_chip_labels.csv")

features = build_features(telemetry)

data = features.merge(
    labels[["chip_id", "triage"]],
    on="chip_id"
)

X = data.drop(columns=["chip_id", "triage"])
y = data["triage"]

# 2. Prepare cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

errors = []

# 3. Train and inspect each fold
for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y),
    start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Find incorrect predictions
    wrong = predictions != y_test.to_numpy()

    for position in range(len(X_test)):
        if wrong[position]:
            original_index = X_test.index[position]

            errors.append({
                "fold": fold,
                "chip_id": data.loc[
                    original_index, "chip_id"
                ],
                "actual": y_test.iloc[position],
                "predicted": predictions[position]
            })

# 4. Display errors
error_df = pd.DataFrame(errors)

print("\nMISCLASSIFIED CHIPS")

if error_df.empty:
    print("No misclassified chips found.")
else:
    print(error_df.to_string(index=False))

    print("\nERROR COUNTS BY ACTUAL CLASS")
    print(error_df["actual"].value_counts())

    print("\nERROR COUNTS BY ACTUAL AND PREDICTED CLASS")
    print(
        error_df.groupby(
            ["actual", "predicted"]
        ).size()
    )

    error_df.to_csv(
        "misclassified_chips.csv",
        index=False
    )

    print("\nSaved misclassified_chips.csv")