# ai_driven_burnin_test

# Zenith — AI-Driven Burn-in Testing

An AI-driven semiconductor burn-in analytics prototype that combines supervised classification and unsupervised anomaly detection to assess chip risk from burn-in telemetry.

## Overview

Traditional burn-in testing evaluates semiconductor reliability by operating chips under controlled stress conditions and monitoring their behaviour.

Zenith analyses chip-level telemetry to identify risk categories and unusual measurement patterns.

The prototype combines two machine learning models:

* **Random Forest:** Classifies each chip into GREEN, AMBER, or RED risk categories.
* **Isolation Forest:** Identifies unusual chip behaviour using anomaly detection.

The outputs are combined into a recommendation to help prioritize chip reviews.

## Features

* Chip-level feature engineering from burn-in telemetry.
* Supervised risk classification.
* Unsupervised anomaly detection.
* Combined chip recommendations.
* Interactive Streamlit dashboard.
* Model evaluation and cross-validation results.
* CSV export of chip-level predictions.
* Docker-based application execution.

## Architecture

```text
Burn-in Telemetry
       |
       v
Feature Engineering
       |
       v
  +------------------+
  |                  |
  v                  v
Random Forest    Isolation Forest
  |                  |
  v                  v
Risk Category    Anomaly Status
  |                  |
  +--------+---------+
           |
           v
   Recommendation
           |
           v
  Prediction Results
           |
           v
 Streamlit Dashboard
```

## Technology Stack

* Python
* Pandas and NumPy
* Scikit-learn
* Joblib
* Plotly
* Streamlit
* Docker and Docker Compose

## Project Structure

```text
ai_driven_burnin_test/
├── app/
│   └── dashboard.py
├── data/
│   ├── zenith_burnin_telemetry.csv
│   └── zenith_chip_labels.csv
├── src/
│   ├── features.py
│   ├── inference.py
│   ├── predict.py
│   └── train.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py
├── zenith_model.pkl
├── zenith_anomaly_model.pkl
└── README.md
```

Additional evaluation scripts and generated output files may also be present.

## Setup and Execution

### Prerequisites

* Python 3.12
* Git
* Docker Desktop (for containerized execution)

### Run locally

Clone the repository and enter its directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd ai_driven_burnin_test
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the prediction pipeline:

```powershell
python src/predict.py
```

This generates `chip_predictions.csv` in the project root.

### Launch the dashboard locally

```powershell
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

Alternatively, run the complete application:

```powershell
python run.py
```

### Run with Docker

Build and launch the application:

```powershell
docker compose up --build
```

Open `http://localhost:8501`.

Stop the application with:

```powershell
docker compose down
```

## Understanding the Recommendations

| Recommendation     | Meaning                                                                           |
| ------------------ | --------------------------------------------------------------------------------- |
| `PASS_CANDIDATE`   | The model predicts GREEN and the anomaly detector does not flag the chip.         |
| `REVIEW`           | The predicted category is AMBER, or the anomaly detector flags unusual behaviour. |
| `REJECT_CANDIDATE` | The model predicts RED.                                                           |

These are model-generated screening recommendations, not definitive proof of physical chip reliability.

## Model Evaluation

The project includes model evaluation and cross-validation scripts to assess classification performance.

Evaluation on training data should not be interpreted as an independent estimate of performance on unseen chips.

Because the current dataset is synthetic, the reported results do not establish real-world semiconductor reliability or production readiness.

## Limitations

* The current dataset is synthetic.
* Predictions depend on the quality and representativeness of the input telemetry.
* Anomaly detection identifies unusual patterns, which do not necessarily indicate physical defects.
* Recommendations require appropriate engineering validation before being used in real manufacturing decisions.

## Future Improvements

* Validate the models on real semiconductor burn-in datasets.
* Improve input validation and automated testing.
* Add model and data versioning.
* Deploy the inference service independently if scaling requirements justify it.
* Integrate monitoring and model-drift detection.

## Disclaimer

Zenith is an experimental analytics prototype intended for demonstration and research. It is not a certified semiconductor testing or production-release system.
