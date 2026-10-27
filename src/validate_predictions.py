import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# 1. Load actual labels
actual = pd.read_csv("data/zenith_chip_labels.csv")

# 2. Load model predictions
predictions = pd.read_csv("chip_predictions.csv")

# 3. Combine actual labels with predictions using chip_id
comparison = actual[["chip_id", "triage"]].merge(
    predictions[["chip_id", "predicted_triage"]],
    on="chip_id",
    how="inner",
)

# 4. Compare actual and predicted labels
y_true = comparison["triage"]
y_pred = comparison["predicted_triage"]

print("\nNUMBER OF CHIPS COMPARED:")
print(len(comparison))

print("\nACCURACY:")
print(round(accuracy_score(y_true, y_pred), 4))

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=["GREEN", "AMBER", "RED"],
        zero_division=0,
    )
)

print("\nCONFUSION MATRIX")
print("Class order: GREEN, AMBER, RED")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=["GREEN", "AMBER", "RED"],
    )
)

# 5. Save individual prediction errors
comparison["correct"] = y_true == y_pred

errors = comparison[~comparison["correct"]]

errors.to_csv("prediction_errors.csv", index=False)

print("\nINCORRECT PREDICTIONS:", len(errors))
print("Errors saved to prediction_errors.csv")
