# 30-Day Hospital Readmission Risk Prediction

A machine learning project predicting early hospital readmission risk using 
10 years of clinical data from 130 US hospitals. Built with Python, pandas, 
and scikit-learn.

## Overview

Hospital readmissions within 30 days are a major quality metric in healthcare 
and a key focus of value-based care models. As a Certified Nursing Assistant 
with 3 years of clinical experience, I built this model to explore which 
patient factors most strongly predict early readmission risk.

## Dataset

- **Source:** Diabetes 130-US Hospitals dataset (UCI / Kaggle)
- **Records:** 101,766 patient encounters across 130 US hospitals (1999-2008)
- **Features:** 50 clinical variables including demographics, diagnoses, 
  medications, and procedure counts

## Key Findings

- 11.2% early readmission rate (<30 days), reflecting real-world class imbalance
- Top predictors: number of lab procedures and medications — consistent with 
  clinical intuition that higher-intervention patients carry more readmission risk
- Logistic Regression ROC-AUC: 0.64 | Random Forest ROC-AUC: 0.63
- Class imbalance identified as a key challenge — a next step would be applying 
  SMOTE oversampling to improve recall on high-risk patients

## Tech Stack

- **Python** — core analysis
- **pandas / numpy** — data cleaning and feature engineering
- **scikit-learn** — Logistic Regression and Random Forest models
- **matplotlib** — data visualization

## How to Run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Add `diabetic_data.csv` to the project folder
4. Run `readmission_risk.ipynb` in Jupyter or VS Code
