# Zenith — AI-Driven Burn-in Testing

An AI-driven semiconductor burn-in analytics prototype that combines supervised machine learning and unsupervised anomaly detection to assess chip risk from burn-in telemetry.

## Overview

Traditional burn-in testing evaluates semiconductor reliability by operating chips under controlled stress conditions and monitoring their behaviour.

Zenith explores how machine learning can assist this process by analysing telemetry, extracting useful features, classifying chip risk, and identifying unusual behaviour that may warrant further investigation.

The project combines a Random Forest classifier with an Isolation Forest anomaly detector and presents the results through an interactive Streamlit dashboard.

> **Project status:** Research and development prototype using a synthetic burn-in dataset. The results do not establish real-world semiconductor reliability or production readiness.

## Key Features

* **Feature engineering:** Converts time-series telemetry into chip-level features.
* **Supervised classification:** Predicts chip risk categories using a trained Random Forest model.
* **Anomaly detection:** Uses Isolation Forest to identify unusual telemetry patterns.
* **Risk-based recommendations:** Produces candidate actions such as `PASS_CANDIDATE`, `REVIEW`, and `REJECT_CANDIDATE`.
* **Model evaluation:** Includes classification metrics, cross-validation, feature-importance analysis, and error investigation.
* **Interactive dashboard:** Explores chip telemetry, predictions, model performance, and feature behaviour.
* **Docker support:** Packages the application and its dependencies for a reproducible runtime environment.

## System Architecture

text
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
        +------+------+
        |             |
        v             v
 Random Forest   Isolation Forest
 Classifier      Anomaly Detector
        |             |
        +------+------+
               |
               v
       Risk and Anomaly Results
               |
               v
      Candidate Recommendations
               |
        +------+------+
        |             |
        v             v
   CSV Predictions  Streamlit Dashboard


## Project Structure

text
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


The repository also contains supporting analysis scripts, evaluation outputs, CSV reports, and plots. The tree above highlights the main application components rather than every generated artifact.

## Dataset

The project uses two CSV files:

| File                          | Purpose                                                       |
| ----------------------------- | ------------------------------------------------------------- |
| `zenith_burnin_telemetry.csv` | Time-series burn-in telemetry for individual chips            |
| `zenith_chip_labels.csv`      | Chip-level labels used for supervised learning and evaluation |

The synthetic dataset contains 33,800 telemetry records representing 200 chips.

The chip-label distribution is:

| Risk category | Number of chips |
| ------------- | --------------: |
| GREEN         |             150 |
| AMBER         |              20 |
| RED           |              30 |
| **Total**     |         **200** |

The telemetry is aggregated into chip-level features before being passed to the models.

## Machine Learning Approach

### 1. Feature Engineering

The feature-engineering pipeline transforms telemetry into chip-level features, including temporal changes and measurements of chip behaviour over the burn-in period.

The resulting feature matrix contains 200 chip-level rows and 54 columns, including the chip identifier.

### 2. Supervised Classification — Random Forest

The Random Forest classifier predicts one of three risk categories:

* `GREEN`
* `AMBER`
* `RED`

The model uses class weighting to account for differences in the number of examples in each category.

### 3. Unsupervised Anomaly Detection — Isolation Forest

Isolation Forest identifies chips with unusual feature patterns.

An anomaly is an observation that differs from the patterns learned by the model. An anomaly is not automatically proof that a chip is defective.

### 4. Candidate Recommendations

The inference pipeline combines classification and anomaly results to generate candidate recommendations:

| Condition                                | Recommendation     |
| ---------------------------------------- | ------------------ |
| RED classification                       | `REJECT_CANDIDATE` |
| AMBER classification or detected anomaly | `REVIEW`           |
| Otherwise                                | `PASS_CANDIDATE`   |

These are prototype recommendations, not validated manufacturing decisions. Real deployment would require engineering review, representative production data, and validation against established test procedures.

## Model Evaluation

The current Random Forest holdout evaluation produced the following results:

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |  97.0% |
| Balanced accuracy | 91.67% |
| GREEN F1-score    |   0.98 |
| AMBER F1-score    |   0.86 |
| RED F1-score      |   1.00 |

Five-fold cross-validation produced:

| Metric            |  Mean | Standard deviation |
| ----------------- | ----: | -----------------: |
| Accuracy          | 0.955 |              0.037 |
| Balanced accuracy | 0.867 |              0.109 |
| Macro F1-score    | 0.890 |              0.090 |

These results are specific to the current synthetic dataset and experimental setup. They should not be interpreted as expected performance on real semiconductor manufacturing data.

The repository also includes scripts and artifacts for model comparison, feature importance, cross-validation, and prediction-error analysis.

## Streamlit Dashboard

The dashboard provides interactive views for:

* Overview
* Feature Analysis
* Chip Explorer
* Model Predictions
* Model Evaluation
* Cross-Validation Results

## Getting Started

### Prerequisites

* Python 3.12 recommended
* Git
* Docker Desktop (optional, for containerized execution)

### 1. Clone the Repository

git clone https://github.com/zaid168/ai_driven_burnin_test.git
cd ai_driven_burnin_test


### 2. Create a Virtual Environment

**Windows PowerShell:**


python -m venv .venv
.\.venv\Scripts\Activate.ps1


If PowerShell blocks environment activation, you can invoke the environment's Python directly instead.

### 3. Install Dependencies


python -m pip install -r requirements.txt


### 4. Run the Application

From the project root:


python run.py


Once the application starts, open:

text
http://localhost:8501


The launcher runs the prediction pipeline and starts the Streamlit dashboard.

## Running with Docker

Make sure Docker Desktop is running, then execute these commands from the project root:


docker compose up --build


Open the dashboard at:

text
http://localhost:8501


To stop the application:


docker compose down


## Reproducibility

The repository includes trained model artifacts so inference can be run without retraining the models.

To retrain or regenerate results, review the scripts under `src/` and confirm their input paths and output behaviour before execution.

## Limitations and Future Work

* The current dataset is synthetic and may not capture the complexity of real semiconductor failure behaviour.
* Model performance requires validation using independent, representative data.
* Anomaly detection identifies unusual patterns, not confirmed defects.
* Candidate recommendations require validation against manufacturing requirements.
* Future work could explore additional telemetry features, improved temporal modelling, explainability, and evaluation on real or appropriately simulated production data.

## Disclaimer

Zenith is an experimental analytics prototype intended for educational and research purposes. It is not a certified semiconductor testing system and should not be used as the sole basis for manufacturing, quality-control, or reliability decisions.
