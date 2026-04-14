# Customer Churn Prediction

## Overview
End-to-End Machine Learning project to predict customer churn
using Azure Machine Learning and scikit-learn.

## Business Problem
A telecom company wants to identify customers who are likely 
to cancel their subscription before they actually do.

## Results
- Best Model: Gradient Boosting
- Accuracy: 80.27%
- AUC: 85.17%

## Project Structure
customer-churn-prediction/
├── data/                    # Data files (not tracked)
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   ├── 02_Featurization.ipynb # Data preparation
│   ├── 03_Training.ipynb    # Model training
│   └── 04_Evaluation.ipynb  # Model evaluation
├── src/                     # Python scripts
├── azure-ml/                # Azure ML configurations
├── .gitignore
├── requirements.txt
└── README.md

## Technologies
- Python
- scikit-learn
- pandas
- matplotlib
- Azure Machine Learning
- MLflow

## How to Run
1. Clone the repository
2. Install requirements: pip install -r requirements.txt
3. Download dataset from Kaggle: 
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
4. Place dataset in data/ folder
5. Run notebooks in order (01 → 02 → 03 → 04)