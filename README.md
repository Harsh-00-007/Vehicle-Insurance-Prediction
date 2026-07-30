# 🚗 Vehicle Insurance Prediction — End-to-End MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-FF9900.svg)](https://aws.amazon.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248.svg)](https://www.mongodb.com/atlas)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A production-style machine learning system that predicts whether a health insurance customer is likely to be interested in purchasing **vehicle insurance** — built as a complete MLOps pipeline, not just a notebook. Data flows from MongoDB Atlas through validation, transformation, training, and evaluation stages, with the winning model automatically versioned to AWS S3 and served through a FastAPI web app that is containerized and deployed via a GitHub Actions CI/CD pipeline to AWS EC2.

**🔗 Live demo:** [http://3.104.109.151:5000](http://3.104.109.151:5000)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Model Details](#model-details)
- [API Endpoints](#api-endpoints)
- [Running Locally](#running-locally)
- [Environment Variables](#environment-variables)
- [Deployment (CI/CD)](#deployment-cicd)
- [Dataset](#dataset)
- [License](#license)
- [Author](#author)

---

## Overview

Insurance companies want to know one thing before they spend money on marketing: **which existing health insurance customers are actually likely to buy vehicle insurance?** This project builds a binary classification model to answer that question, then wraps it in the kind of infrastructure a real ML team would use to keep it running in production — versioned data, a repeatable training pipeline, model registry, and automated deployment.

Rather than a one-off script, the project is organized as a set of independent, testable pipeline **components** (ingestion → validation → transformation → training → evaluation → pushing), each producing a typed **artifact** that feeds the next stage. This makes the pipeline easy to debug, retrain, and extend.

## Architecture

```mermaid
flowchart LR
    A[MongoDB Atlas<br/>Proj1-Data collection] --> B[Data Ingestion<br/>train/test split]
    B --> C[Data Validation<br/>schema.yaml checks]
    C --> D[Data Transformation<br/>encoding · scaling · SMOTEENN]
    D --> E[Model Trainer<br/>RandomForestClassifier]
    E --> F[Model Evaluation<br/>champion vs challenger F1]
    F -->|accepted| G[Model Pusher<br/>AWS S3 model registry]
    G --> H[FastAPI App<br/>prediction_pipeline.py]
    H --> I[Docker Image]
    I --> J[GitHub Actions<br/>Build & Push to ECR]
    J --> K[Self-hosted Runner on EC2<br/>docker run -p 5000:5000]
    K --> L((Live App<br/>3.104.109.151:5000))
```

Every stage is config-driven (`config/schema.yaml`, environment-based secrets) and communicates through strongly-typed artifact/config entities, rather than passing raw dataframes around — the same pattern used in production ML systems.

## Tech Stack

| Layer                  | Technology                                              |
|-------------------------|----------------------------------------------------------|
| Language                | Python 3.10                                              |
| Data storage            | MongoDB Atlas                                             |
| ML / Data processing    | scikit-learn, pandas, NumPy, imbalanced-learn (SMOTEENN)  |
| Model                   | Random Forest Classifier                                  |
| Backend / API           | FastAPI, Uvicorn, Jinja2                                   |
| Model registry / storage| AWS S3 (via boto3)                                        |
| Containerization        | Docker                                                     |
| CI/CD                   | GitHub Actions (build → ECR → self-hosted EC2 runner)      |
| Cloud                   | AWS (S3, ECR, EC2, IAM)                                    |

## Project Structure

```
Vehicle-Insurance-Prediction/
├── app.py                        # FastAPI app: form UI, /train and / (predict) routes
├── demo.py                       # Quick script to trigger the training pipeline directly
├── Dockerfile                    # Container spec, exposes port 5000
├── requirements.txt
├── setup.py / pyproject.toml     # Packaging for the local `src` module
├── config/
│   ├── schema.yaml               # Column names, dtypes, categorical/numerical splits
│   └── model.yaml
├── notebook/
│   ├── analysis.ipynb            # EDA & experimentation
│   └── mongoDB.ipynb             # Notebook used to push the raw dataset into MongoDB
├── static/css/                   # Styling for the web form
├── templates/
│   └── vehicledata.html          # Input form + prediction result page
├── .github/workflows/aws.yaml    # CI/CD pipeline definition
└── src/
    ├── components/                # Pipeline stages
    │   ├── data_ingestion.py
    │   ├── data_validation.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    │   ├── model_evaluation.py
    │   └── model_pusher.py
    ├── entity/                    # Config & artifact dataclasses, estimator wrappers
    ├── pipline/                   # training_pipeline.py & prediction_pipeline.py
    ├── data_access/                # MongoDB read layer
    ├── cloud_storage/              # AWS S3 read/write layer
    ├── configuration/               # MongoDB & AWS client setup
    ├── constants/                   # Central config (paths, hyperparameters, bucket names)
    ├── exception/                    # Custom exception handling
    └── logger/                        # Central logging setup
```

## ML Pipeline

1. **Data Ingestion** — Pulls the raw collection from MongoDB Atlas, exports it to a feature store CSV, then performs a train/test split.
2. **Data Validation** — Validates incoming data against `config/schema.yaml` (expected columns, dtypes, categorical vs. numerical features) before it's allowed further into the pipeline.
3. **Data Transformation** — Encodes categorical fields (`Gender`, `Vehicle_Age`, `Vehicle_Damage`), scales `Annual_Premium` with `MinMaxScaler`, and — since interested customers are a minority class — rebalances the training data with **SMOTEENN** (combined over/under-sampling).
4. **Model Trainer** — Trains a `RandomForestClassifier` and computes accuracy, F1, precision, and recall on the held-out test set. Training only proceeds past a configured minimum accuracy threshold.
5. **Model Evaluation** — Compares the newly trained model's F1 score against the current production model already sitting in S3 (if one exists) and only marks the new model as "accepted" if it's a genuine improvement — a simple champion/challenger pattern.
6. **Model Pusher** — If accepted, bundles the trained model with its preprocessing object and uploads it to an S3-backed model registry, making it immediately available to the serving app.

Training can be re-triggered at any time — either by running `demo.py` locally, or by hitting the **`/train`** endpoint on the live app itself.

## Model Details

- **Algorithm:** Random Forest Classifier (`n_estimators=200`, `max_depth=10`, `criterion="entropy"`, tuned `min_samples_split` / `min_samples_leaf`)
- **Class imbalance handling:** SMOTEENN on the training split
- **Evaluation metrics:** Accuracy, F1, Precision, Recall
- **Promotion rule:** A newly trained model is only pushed to the registry if its F1 score beats the currently deployed model by a configured margin

## API Endpoints

| Method | Route     | Description                                                        |
|--------|-----------|----------------------------------------------------------------------|
| GET    | `/`       | Renders the vehicle data input form                                  |
| POST   | `/`       | Accepts form input, runs it through the deployed model, returns `Response-Yes` / `Response-No` |
| GET    | `/train`  | Triggers the full training pipeline end-to-end from the live app     |

## Running Locally

```bash
# Clone the repo
git clone https://github.com/Harsh-00-007/Vehicle-Insurance-Prediction.git
cd Vehicle-Insurance-Prediction

# Install dependencies
pip install -r requirements.txt

# Set required environment variables (see below), then run
python app.py
```

The app will start on `http://0.0.0.0:5000`.

Or, with Docker:

```bash
docker build -t vehicle-insurance-prediction .
docker run -p 5000:5000 \
  -e MONGODB_URL="<your-mongodb-connection-string>" \
  -e AWS_ACCESS_KEY_ID="<your-key>" \
  -e AWS_SECRET_ACCESS_KEY="<your-secret>" \
  -e AWS_DEFAULT_REGION="ap-southeast-2" \
  vehicle-insurance-prediction
```

## Environment Variables

| Variable                | Purpose                                   |
|--------------------------|---------------------------------------------|
| `MONGODB_URL`             | Connection string for MongoDB Atlas          |
| `AWS_ACCESS_KEY_ID`       | AWS IAM credentials for S3 access            |
| `AWS_SECRET_ACCESS_KEY`   | AWS IAM credentials for S3 access            |
| `AWS_DEFAULT_REGION`      | AWS region (`ap-southeast-2`)                |

## Deployment (CI/CD)

Every push to `main` triggers a GitHub Actions workflow (`.github/workflows/aws.yaml`) that:

1. **Builds** the Docker image and **pushes** it to a private Amazon ECR repository.
2. Hands off to a **self-hosted runner** (an EC2 instance) which pulls the freshly built image and runs it with the required secrets injected as environment variables, exposing the app on port `5000`.

This means shipping a model or code update is literally a `git push` — no manual server access required.

## Dataset

The pipeline is built around the well-known **Health Insurance Cross-Sell Prediction** dataset (~381K records) — customer demographic, policy, and vehicle attributes (`Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, `Vintage`) used to predict `Response`: whether the customer is interested in vehicle insurance.

## License

This project is licensed under the [MIT License](./LICENSE).

## Author

**Harsh Gupta**
GitHub: [@Harsh-00-007](https://github.com/Harsh-00-007)

If you find this project useful, consider giving the repo a ⭐!