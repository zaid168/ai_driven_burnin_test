import sys
from pathlib import Path
import joblib

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow Python to import our existing feature engineering module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Zenith | Chip Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("Zenith | Burn-in Analytics")
st.caption(
    "Explore chip measurements, compare classes, "
    "and investigate unusual behaviour."
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    telemetry = pd.read_csv(
        ROOT / "data" / "zenith_burnin_telemetry.csv"
    )

    labels = pd.read_csv(
        ROOT / "data" / "zenith_chip_labels.csv"
    )

    features = build_features(telemetry)

    data = features.merge(
        labels[["chip_id", "triage", "failure_mode"]],
        on="chip_id",
        how="left"
    )

    return telemetry, data


try:
    telemetry, data = load_data()

except Exception as error:
    st.error(f"Could not load project data: {error}")
    st.stop()

@st.cache_resource
def load_models():
    risk_model = joblib.load(ROOT / "zenith_model.pkl")
    anomaly_model = joblib.load(ROOT / "zenith_anomaly_model.pkl")
    return risk_model, anomaly_model


@st.cache_data

@st.cache_data
def generate_predictions():
    risk_model, anomaly_model = load_models()

    telemetry = pd.read_csv(
        ROOT / "data" / "zenith_burnin_telemetry.csv"
    )

    features = build_features(telemetry)

    # Random Forest expects 53 features
    risk_columns = list(risk_model.feature_names_in_)
    X_risk = features[risk_columns]

    # Isolation Forest expects 29 features
    anomaly_columns = list(anomaly_model.feature_names_in_)
    X_anomaly = features[anomaly_columns]

    # Generate predictions using the correct feature sets
    results = features[["chip_id"]].copy()

    results["predicted_triage"] = risk_model.predict(X_risk)

    results["anomaly_prediction"] = anomaly_model.predict(X_anomaly)
    results["anomaly_score"] = anomaly_model.decision_function(
        X_anomaly
    )

    results["anomaly_status"] = results["anomaly_prediction"].map({
        1: "NORMAL",
        -1: "ANOMALY"
    })

    # Include important measurements
    measurement_columns = [
        "max_leakage",
        "leakage_change",
        "max_ddq",
        "delay_change"
    ]

    for column in measurement_columns:
        results[column] = features[column].values

    # Combine the model outputs
    def make_recommendation(row):
        if row["predicted_triage"] == "RED":
            return "REJECT_CANDIDATE"

        if (
            row["predicted_triage"] == "AMBER"
            or row["anomaly_status"] == "ANOMALY"
        ):
            return "REVIEW"

        return "PASS_CANDIDATE"

    results["recommendation"] = results.apply(
        make_recommendation,
        axis=1
    )

    return results


# --------------------------------------------------
# 3. SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Feature Analysis",
        "Chip Explorer",
        "Model Predictions",
        "Model Evaluation",
        "Cross-Validation Results"
    ]
)


# --------------------------------------------------
# 4. OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.subheader("Dataset Overview")

    total_chips = data["chip_id"].nunique()
    total_records = len(telemetry)

    green_count = (data["triage"] == "GREEN").sum()
    amber_count = (data["triage"] == "AMBER").sum()
    red_count = (data["triage"] == "RED").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Chips", total_chips)
    col2.metric("GREEN", green_count)
    col3.metric("AMBER", amber_count)
    col4.metric("RED", red_count)

    st.divider()

    st.subheader("Triage Distribution")

    counts = (
        data["triage"]
        .value_counts()
        .rename_axis("triage")
        .reset_index(name="chip_count")
    )

    fig = px.bar(
        counts,
        x="triage",
        y="chip_count",
        color="triage",
        text="chip_count",
        title="Number of Chips in Each Class",
        labels={
            "triage": "Actual Class",
            "chip_count": "Number of Chips"
        }
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(dtick=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"The telemetry dataset contains {total_records:,} "
        f"measurement records across {total_chips} chips."
    )


# --------------------------------------------------
# 5. FEATURE ANALYSIS
# --------------------------------------------------

elif page == "Feature Analysis":

    st.subheader("Compare Features Between Classes")

    st.write(
        "Select a measurement to compare its distribution "
        "between GREEN and AMBER chips."
    )

    selected_features = [
        "delay_change",
        "max_ddq",
        "max_leakage",
        "max_lot_delay_z",
        "window_leakage_change_mid1",
        "window_leakage_change_mid2",
        "window_leakage_change_late",
        "window_max_ddq_mid1",
        "window_max_ddq_mid2",
        "window_max_leakage_z_late"
    ]

    available_features = [
        feature
        for feature in selected_features
        if feature in data.columns
    ]

    feature = st.selectbox(
        "Choose a feature",
        available_features
    )

    class_filter = st.multiselect(
        "Choose classes",
        ["GREEN", "AMBER", "RED"],
        default=["GREEN", "AMBER"]
    )

    if not class_filter:
        st.warning("Select at least one class.")
        st.stop()

    comparison = data[
        data["triage"].isin(class_filter)
    ].copy()

    # Numerical summary
    summary = (
        comparison.groupby("triage")[feature]
        .agg(
            count="count",
            minimum="min",
            q1=lambda x: x.quantile(0.25),
            median="median",
            q3=lambda x: x.quantile(0.75),
            maximum="max",
            mean="mean"
        )
        .round(5)
    )

    st.markdown("### Numerical Summary")

    st.caption(
        "All values below are calculated directly from the dataset."
    )

    st.dataframe(
        summary,
        use_container_width=True
    )

    # Interactive distribution plot
    st.markdown("### Feature Distribution")

    fig = px.box(
        comparison,
        x="triage",
        y=feature,
        color="triage",
        points="all",
        hover_data=["chip_id"],
        category_orders={
            "triage": ["GREEN", "AMBER", "RED"]
        },
        title=f"{feature}: Comparison Between Classes",
        labels={
            "triage": "Actual Class",
            feature: feature
        }
    )

    fig.update_layout(
        showlegend=False,
        height=550,
        xaxis_title="Actual Chip Class",
        yaxis_title=feature
    )

    fig.update_yaxes(
        tickformat=".5f",
        showgrid=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Each dot represents one chip. Hover over a dot to see "
        "its chip ID and measurement value. Use the graph toolbar "
        "to zoom, pan, or reset the view."
    )

    # Show actual chip-level values
    st.markdown("### Individual Chip Measurements")

    show_all = st.checkbox(
        "Show all individual chip measurements",
        value=False
    )

    if show_all:
        columns = ["chip_id", "triage", feature]

        st.dataframe(
            comparison[columns].sort_values(feature),
            use_container_width=True
        )


# --------------------------------------------------
# 6. CHIP EXPLORER
# --------------------------------------------------

elif page == "Chip Explorer":

    st.subheader("Explore an Individual Chip")

    chip_ids = sorted(data["chip_id"].dropna().unique())

    selected_chip = st.selectbox(
        "Select a chip",
        chip_ids
    )

    chip = data[
        data["chip_id"] == selected_chip
    ].iloc[0]

    col1, col2 = st.columns(2)

    col1.metric(
        "Actual Triage",
        str(chip["triage"])
    )

    col2.metric(
        "Failure Mode",
        str(chip["failure_mode"])
    )

    st.markdown("### Chip Features")

    feature_columns = [
        column
        for column in data.columns
        if column != "chip_id"
        and pd.api.types.is_numeric_dtype(data[column])
    ]

    chip_features = pd.DataFrame({
        "Feature": feature_columns,
        "Value": [
            chip[column]
            for column in feature_columns
        ]
    })

    st.dataframe(
        chip_features.round(5),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Measurements Over Burn-in Time")

    chip_telemetry = telemetry[
        telemetry["chip_id"] == selected_chip
    ].sort_values("burn_in_hour")

    measurement = st.selectbox(
        "Select a measurement",
        [
            "I_leak_uA",
            "I_DDQ_uA",
            "V_th_V",
            "t_pd_ns",
            "chamber_temp_C"
        ]
    )

    fig = px.line(
        chip_telemetry,
        x="burn_in_hour",
        y=measurement,
        markers=True,
        title=f"{selected_chip}: {measurement} Over Time",
        labels={
            "burn_in_hour": "Burn-in Hour",
            measurement: measurement
        }
    )

    fig.update_layout(height=500)

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    # --------------------------------------------------
# 7. MODEL PREDICTIONS
# --------------------------------------------------

elif page == "Model Predictions":

    st.subheader("Model Predictions")
    st.write(
        "Combined outputs from Random Forest and Isolation Forest."
    )

    try:
        predictions = generate_predictions()

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Chips",
        len(predictions)
    )

    col2.metric(
        "Pass Candidates",
        (predictions["recommendation"] == "PASS_CANDIDATE").sum()
    )

    col3.metric(
        "Review Required",
        (predictions["recommendation"] == "REVIEW").sum()
    )

    col4.metric(
        "Reject Candidates",
        (predictions["recommendation"] == "REJECT_CANDIDATE").sum()
    )

    st.divider()

    # Filters
    selected_triage = st.multiselect(
        "Filter by predicted triage",
        ["GREEN", "AMBER", "RED"],
        default=["GREEN", "AMBER", "RED"]
    )

    selected_anomaly = st.multiselect(
        "Filter by anomaly status",
        ["NORMAL", "ANOMALY"],
        default=["NORMAL", "ANOMALY"]
    )

    filtered = predictions[
        predictions["predicted_triage"].isin(selected_triage)
        & predictions["anomaly_status"].isin(selected_anomaly)
    ]

    st.subheader("Chip-Level Results")

    st.dataframe(
        filtered.sort_values(
            ["predicted_triage", "anomaly_score"]
        ),
        use_container_width=True,
        hide_index=True
    )

    # Download
    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="chip_predictions.csv",
        mime="text/csv"
    )

    # Inspect a specific chip
    st.divider()
    st.subheader("Inspect a Chip")

    if not filtered.empty:
        selected_chip = st.selectbox(
            "Choose a chip",
            filtered["chip_id"].tolist()
        )

        chip = filtered[
            filtered["chip_id"] == selected_chip
        ].iloc[0]

        st.write(f"**Predicted triage:** {chip['predicted_triage']}")
        st.write(f"**Anomaly status:** {chip['anomaly_status']}")
        st.write(f"**Recommendation:** {chip['recommendation']}")
        st.write(f"**Anomaly score:** {chip['anomaly_score']:.5f}")

        st.markdown("#### Measurements")

        st.dataframe(
            chip[
                [
                    "max_leakage",
                    "leakage_change",
                    "max_ddq",
                    "delay_change"
                ]
            ].to_frame(name="Value"),
            use_container_width=True
        )

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix
    )

    import plotly.figure_factory as ff

    st.subheader("Model Evaluation")

    st.write(
        "Compare Random Forest predictions with the actual "
        "triage labels in the dataset."
    )

    try:
        predictions = generate_predictions()

        # Load actual labels
        labels = pd.read_csv(
            ROOT / "data" / "zenith_chip_labels.csv"
        )

        evaluation = predictions.merge(
            labels[["chip_id", "triage"]],
            on="chip_id",
            how="inner"
        )

    except Exception as error:
        st.error(f"Evaluation failed: {error}")
        st.stop()

    # Actual and predicted labels
    y_true = evaluation["triage"]
    y_pred = evaluation["predicted_triage"]

    classes = ["GREEN", "AMBER", "RED"]

    # Summary metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Balanced Accuracy",
        f"{balanced_accuracy * 100:.2f}%"
    )

    col3.metric(
        "Chips Evaluated",
        len(evaluation)
    )

    st.info(
        "These results evaluate predictions on the same dataset "
        "used during model development. They are not an "
        "independent estimate of performance on unseen chips."
    )

    st.divider()

    # Confusion matrix
    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Number of Chips"
        },
        title="Confusion Matrix"
    )

    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        height=500
    )

    # Put actual GREEN at the top
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual classes. Columns represent "
        "predicted classes. Diagonal cells are correct predictions."
    )

    st.divider()

    # Per-class metrics
    st.markdown("### Per-Class Performance")

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # Misclassified chips
    st.markdown("### Misclassified Chips")

    errors = evaluation[
        evaluation["triage"] != evaluation["predicted_triage"]
    ][
        [
            "chip_id",
            "triage",
            "predicted_triage",
            "anomaly_status",
            "anomaly_score"
        ]
    ].copy()

    errors = errors.rename(
        columns={
            "triage": "actual_class",
            "predicted_triage": "predicted_class"
        }
    )

    st.write(
        f"Incorrect predictions: {len(errors)} "
        f"out of {len(evaluation)} chips."
    )

    st.dataframe(
        errors,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download misclassified chips",
        data=errors.to_csv(index=False).encode("utf-8"),
        file_name="model_evaluation_errors.csv",
        mime="text/csv"
    )
# --------------------------------------------------
# CROSS-VALIDATION RESULTS
# --------------------------------------------------

elif page == "Cross-Validation Results":

    import json

    st.subheader("Cross-Validation Results")

    st.write(
        "Performance of Random Forest across five stratified "
        "train/test splits."
    )

    results_path = ROOT / "cross_validation_results.csv"
    matrices_path = ROOT / "cross_validation_confusion_matrices.json"

    if not results_path.exists() or not matrices_path.exists():
        st.warning(
            "Cross-validation result files were not found. "
            "Run src/evaluate_model.py first."
        )
        st.stop()

    # Load saved evaluation results
    results = pd.read_csv(results_path)

    with open(matrices_path, "r", encoding="utf-8") as file:
        confusion_data = json.load(file)

    # Summary metrics
    accuracy_mean = results["accuracy"].mean()
    balanced_mean = results["balanced_accuracy"].mean()
    f1_mean = results["macro_f1"].mean()

    accuracy_std = results["accuracy"].std(ddof=0)
    balanced_std = results["balanced_accuracy"].std(ddof=0)
    f1_std = results["macro_f1"].std(ddof=0)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mean Accuracy",
        f"{accuracy_mean:.1%}",
        help=f"Standard deviation: {accuracy_std:.1%}"
    )

    col2.metric(
        "Mean Balanced Accuracy",
        f"{balanced_mean:.1%}",
        help=f"Standard deviation: {balanced_std:.1%}"
    )

    col3.metric(
        "Mean Macro F1",
        f"{f1_mean:.1%}",
        help=f"Standard deviation: {f1_std:.1%}"
    )

    st.info(
        "Each fold trains a separate model on 80% of the chips "
        "and evaluates it on the remaining 20%. Every chip is "
        "held out for evaluation once."
    )

    st.divider()

    # Results table
    st.markdown("### Results by Fold")

    display_results = results.copy()

    for column in ["accuracy", "balanced_accuracy", "macro_f1"]:
        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    display_results = display_results.rename(
        columns={
            "fold": "Fold",
            "accuracy": "Accuracy (%)",
            "balanced_accuracy": "Balanced Accuracy (%)",
            "macro_f1": "Macro F1 (%)"
        }
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Visual comparison
    st.markdown("### Performance Across Folds")

    chart_data = results.melt(
        id_vars="fold",
        value_vars=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1"
        ],
        var_name="metric",
        value_name="score"
    )

    chart_data["score"] *= 100

    chart_data["metric"] = chart_data["metric"].map({
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro F1"
    })

    fig = px.bar(
        chart_data,
        x="fold",
        y="score",
        color="metric",
        barmode="group",
        text_auto=True,
        title="Evaluation Score by Fold",
        labels={
            "fold": "Cross-Validation Fold",
            "score": "Score (%)",
            "metric": "Metric"
        }
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
            dtick=1
        ),
        yaxis=dict(
            range=[0, 105],
            dtick=10
        ),
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Individual fold confusion matrix
    st.markdown("### Inspect a Fold's Confusion Matrix")

    selected_fold = st.selectbox(
        "Select fold",
        [item["fold"] for item in confusion_data]
    )

    fold_item = next(
        item for item in confusion_data
        if item["fold"] == selected_fold
    )

    classes = ["GREEN", "AMBER", "RED"]

    cm_df = pd.DataFrame(
        fold_item["matrix"],
        index=classes,
        columns=classes
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Number of Chips"
        },
        title=f"Confusion Matrix — Fold {selected_fold}"
    )

    fig_cm.update_layout(height=450)

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.caption(
        "Rows are actual classes; columns are predicted classes. "
        "Values outside the diagonal represent misclassifications."
    )

    st.download_button(
        "Download cross-validation results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="cross_validation_results.csv",
        mime="text/csv"
    )
