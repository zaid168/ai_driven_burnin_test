# 🔬 Zenith — AI-Driven Semiconductor Burn-in Testing

**Machine Learning · Anomaly Detection · Telemetry Analytics**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker\&logoColor=white)](https://www.docker.com/)

An AI-driven semiconductor burn-in analytics prototype that combines supervised classification and unsupervised anomaly detection to assess chip risk from burn-in telemetry.

---

## Dashboard Preview

### Overview
![Zenith Dashboard Overview](docs/screenshots/overview.png)

### Model Predictions
![Zenith Model Predictions](docs/screenshots/predictions.png)

### Model Evaluation
![Zenith Model Evaluation](docs/screenshots/evaluation.png)

---
## Overview

Traditional semiconductor burn-in testing evaluates chip reliability by operating devices under controlled stress conditions while monitoring their behaviour.

Zenith explores how machine learning can support this process by analysing telemetry, extracting meaningful features, classifying chip risk, and identifying unusual behaviour that may require further investigation.

The system combines a **Random Forest classifier** and an **Isolation Forest anomaly detector**, with an interactive Streamlit dashboard for exploring predictions and model performance.

> **Project status:** Research and development prototype using a synthetic dataset. Its results do not establish real-world semiconductor reliability or production readiness.

## Key Features

* **Telemetry analysis:** Processes time-series burn-in data for individual chips.
* **Feature engineering:** Extracts chip-level features from telemetry measurements.
* **Risk classification:** Predicts GREEN, AMBER, and RED chip categories using Random Forest.
* **Anomaly detection:** Uses Isolation Forest to identify unusual chip behaviour.
* **Candidate recommendations:** Generates PASS_CANDIDATE, REVIEW, and REJECT_CANDIDATE recommendations.
* **Model evaluation:** Includes classification metrics, cross-validation, feature-importance analysis, and prediction-error investigation.
* **Interactive dashboard:** Explores telemetry features, individual chips, predictions, and model evaluation.
* **Docker support:** Runs the application in a containerized environment.

## System Architecture

```text
          Burn-in Telemetry + Chip Labels
                         |
                         v
                  Data Processing
                         |
                         v
                 Feature Engineering
                         |
                         v
                 Chip-Level Features
                         |
                +--------+--------+
                |                 |
                v                 v
          Random Forest     Isolation Forest
           Classifier       Anomaly Detector
                |                 |
                +--------+--------+
                         |
                         v
               Risk and Anomaly Results
                         |
                         v
               Candidate Recommendations
                         |
                +--------+--------+
                |                 |
                v                 v
          CSV Predictions    Streamlit Dashboard
```

## Project Structure

```text
ai_driven_burnin_test/
├── app/
│   └── dashboard.py
├── data/
│   ├── zenith_burnin_telemetry.csv
│   └── zenith_chip_labels.csv
├── src/
│   ├── anomaly_detection.py
│   ├── features.py
│   ├── inference.py
│   ├── predict.py
│   ├── train.py
│   ├── evaluate_model.py
│   ├── compare_models.py
│   ├── compare_classes.py
│   ├── compare_class_weights.py
│   ├── compare_feature_importance.py
│   ├── combined_analysis.py
│   ├── temporal_analysis.py
│   ├── investigate_errors.py
│   ├── investigate_errors.py
│   ├── validate_predictions.py
│   ├── test_pipeline.py
│   ├── anazlyze_errors.py
│   └── plot_class_comparision.py
├── zenith_model.pkl
├── zenith_anomaly_model.pkl
├── chip_predictions.csv
├── run.py
├── requirements.txt
├── requirements-lock.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

The repository also contains supporting analysis scripts, evaluation outputs, CSV reports, and visualization files that are not all listed in the tree above.

## Dataset

The project uses two CSV files.

| File                          | Description                                                   |
| ----------------------------- | ------------------------------------------------------------- |
| `zenith_burnin_telemetry.csv` | Time-series burn-in telemetry for individual chips            |
| `zenith_chip_labels.csv`      | Chip-level labels used for supervised learning and evaluation |

The synthetic dataset contains **33,800 telemetry records representing 200 chips**.

### Risk Category Distribution

| Risk Category | Number of Chips |
| ------------- | --------------: |
| GREEN         |             150 |
| AMBER         |              20 |
| RED           |              30 |
| **Total**     |         **200** |

The telemetry is transformed into chip-level features before being passed to the machine-learning models.

## Machine Learning Approach

### 1. Feature Engineering

The feature-engineering pipeline transforms raw telemetry into chip-level features.

Features include measurements and temporal indicators that describe chip behaviour over the burn-in period.

The resulting feature matrix contains **200 chip-level rows and 54 columns**, including the chip identifier.

### 2. Supervised Classification — Random Forest

A Random Forest classifier predicts one of three risk categories:

* `GREEN`
* `AMBER`
* `RED`

The model uses class weighting to account for differences in the number of samples in each category.

The classification output provides a risk category for each chip based on its engineered features.

### 3. Unsupervised Anomaly Detection — Isolation Forest

Isolation Forest identifies observations with unusual feature patterns.

Unlike the supervised classifier, the anomaly detector identifies unusual behaviour rather than predicting a known risk category.

An anomaly does not automatically indicate a defective chip. It identifies a chip that may warrant additional investigation.

The detector's configured contamination parameter controls its expected anomaly proportion; it should not be interpreted as the actual semiconductor defect rate.

### 4. Candidate Recommendations

The inference pipeline combines classification and anomaly-detection results to produce candidate recommendations.

| Condition                                | Recommendation     |
| ---------------------------------------- | ------------------ |
| RED classification                       | `REJECT_CANDIDATE` |
| AMBER classification or detected anomaly | `REVIEW`           |
| Otherwise                                | `PASS_CANDIDATE`   |

These recommendations are intended for prototype analysis. They are not validated manufacturing decisions and should not replace established testing procedures or engineering review.

## Model Evaluation

The current Random Forest holdout evaluation produced the following results.

### Holdout Performance

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |  97.0% |
| Balanced Accuracy | 91.67% |
| GREEN F1-score    |   0.98 |
| AMBER F1-score    |   0.86 |
| RED F1-score      |   1.00 |

### Five-Fold Cross-Validation

| Metric            |  Mean | Standard Deviation |
| ----------------- | ----: | -----------------: |
| Accuracy          | 0.955 |              0.037 |
| Balanced Accuracy | 0.867 |              0.109 |
| Macro F1-score    | 0.890 |              0.090 |

The repository also contains scripts and artifacts for:

* Model comparison
* Class-weight experiments
* Feature-importance analysis
* Cross-validation
* Misclassified-chip investigation
* Prediction-error analysis
* Temporal feature analysis

> **Important limitation:** These metrics were obtained using the current synthetic dataset and experimental setup. They should not be interpreted as expected performance on real semiconductor manufacturing data. Independent validation on representative data is necessary before drawing conclusions about real-world reliability.

## Streamlit Dashboard

Zenith includes an interactive dashboard built with Streamlit.

The dashboard provides the following views:

* **Overview:** Summary of the dataset and analysis.
* **Feature Analysis:** Exploration of engineered features.
* **Chip Explorer:** Inspection of individual chips and their features.
* **Model Predictions:** Exploration of predicted risk categories and candidate recommendations.
* **Model Evaluation:** Review of classification performance.
* **Cross-Validation Results:** Examination of cross-validation metrics.

## Getting Started

### Prerequisites

* Python 3.12 recommended
* Git
* Docker Desktop (optional, for containerized execution)

### 1. Clone the Repository

```bash
git clone https://github.com/zaid168/ai_driven_burnin_test.git
cd ai_driven_burnin_test
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

A virtual environment isolates the project's Python dependencies from other Python projects.

If PowerShell prevents environment activation, you can use the virtual environment's Python executable directly.

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the Application

From the project root, execute:

```powershell
python run.py
```

The launcher runs the prediction pipeline and starts the Streamlit dashboard.

Open the following address in your browser:

```text
http://localhost:8501
```

## Running with Docker

Docker provides a containerized environment for running the application without manually setting up the project's Python dependencies on the host.

Ensure Docker Desktop is running, then execute these commands from the project root.

### Build and Start

```bash
docker compose up --build
```

Open the dashboard:

```text
http://localhost:8501
```

### Stop the Application

```bash
docker compose down
```

## Model Artifacts

The repository includes saved model artifacts:

| File                       | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `zenith_model.pkl`         | Trained Random Forest classification model |
| `zenith_anomaly_model.pkl` | Trained Isolation Forest anomaly detector  |

These artifacts allow the inference pipeline to use the saved models without retraining them.

Only load model files from trusted sources. Pickle-based model files can execute code when loaded and should not be treated as safe for untrusted inputs.

## Reproducibility

The repository includes:

* Source code for feature engineering and inference
* Scripts for model training and evaluation
* Saved model artifacts
* Dataset files
* Dependency specifications
* Docker configuration
* Generated predictions and analysis outputs

For reproducible experiments, use the provided dependency files and document any changes to the dataset, model configuration, or evaluation procedure.

## Limitations and Future Work

* The dataset is synthetic and may not capture the complexity of real semiconductor failure behaviour.
* Model performance requires validation using independent and representative data.
* Anomaly detection identifies unusual patterns, not confirmed defects.
* Candidate recommendations require validation against manufacturing requirements.
* Synthetic-data performance may not generalize to real-world operating conditions.

Potential future improvements include:

* Validation using representative semiconductor telemetry
* More advanced temporal feature engineering
* Improved model explainability
* Additional anomaly-detection approaches
* Evaluation of false-positive and false-negative costs
* Integration with real burn-in testing workflows

## Disclaimer

Zenith is an experimental analytics prototype intended for educational and research purposes. It is not a certified semiconductor testing system and must not be used as the sole basis for manufacturing, quality-control, or reliability decisions.
