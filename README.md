# Customer Churn Prediction

## Overview
End-to-End Machine Learning project to predict customer churn
using Azure Machine Learning and scikit-learn.

## Business Problem
A telecom company wants to identify customers who are likely
to cancel their subscription before they actually do.

## Results

| Model | Accuracy | AUC | Churn Recall |
|---|---|---|---|
| Gradient Boosting (Baseline) | 80.27% | 85.17% | 50.00% |
| HyperDrive Tuned | 80.20% | 84.79% | 52.40% |
| Threshold 0.3 | 77.50% | 84.55% | 74.60% |
| **HyperDrive + Threshold (Final)** | **72.60%** | **85.17%** | **86.09%** |

## Project Structure
```
customer-churn-prediction/
├── data/                      # Data files (not tracked)
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Featurization.ipynb # Data preparation
│   ├── 03_Training.ipynb      # Model training
│   └── 04_Evaluation.ipynb    # Model evaluation
├── src/
│   ├── prep.py                # Data preparation script
│   ├── train.py               # Training script with MLflow
│   └── evaluate.py            # Evaluation script
├── azure-ml/
│   ├── command_job.yml        # Single training job
│   ├── hyperdrive_job.yml     # Hyperparameter tuning
│   ├── pipeline.yml           # ML Pipeline
│   ├── endpoint.yml           # Real-time endpoint
│   └── deployment.yml         # Model deployment
├── .gitignore
├── requirements.txt
└── README.md
```

## Azure ML Pipeline
```
churn-data → prep → train → (evaluate)
```

## Technologies
- Python
- scikit-learn
- pandas
- matplotlib
- Azure Machine Learning
- MLflow

## How to Run
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download dataset from Kaggle:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
4. Place dataset in `data/` folder
5. Run notebooks in order (01 → 02 → 03 → 04)

## Azure ML Setup
```bash
# 1. Workspace erstellen
az ml workspace create --name aml-churn-prediction \
  --resource-group rg-churn-prediction --location westeurope

# 2. Compute Cluster erstellen
az ml compute create --name cpu-cluster --type AmlCompute \
  --min-instances 0 --max-instances 2 --size Standard_DS2_v2 \
  --workspace-name aml-churn-prediction \
  --resource-group rg-churn-prediction

# 3. Daten hochladen
az ml data create --name churn-data --type uri_folder \
  --path ./data --workspace-name aml-churn-prediction \
  --resource-group rg-churn-prediction

# 4. Pipeline starten
az ml job create --file azure-ml/pipeline.yml \
  --workspace-name aml-churn-prediction \
  --resource-group rg-churn-prediction
```