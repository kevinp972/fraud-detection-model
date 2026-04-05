# Real-Time Fraud Detection Model

## Start Here

For a complete walkthrough of the methodology, findings, and results, please refer to the report:

**[Real Time Fraud Detection Model Report.pdf](Real_Time_Fraud_Detection_Model_Report.pdf)**

The report covers data exploration, feature engineering, model selection, performance evaluation, and financial impact analysis in full detail.

---

## Project Overview

This project builds a transaction-level fraud detection model on corporate card data. The pipeline spans exploratory analysis through model deployment, with the goal of maximizing the Fraud Detection Rate (FDR) — the share of fraudulent transactions captured when reviewing the top N% of highest-scoring transactions.

## Repository Structure

| Notebook | Description |
|---|---|
| `1_data_exploration.ipynb` | Exploratory analysis of raw transaction data |
| `2_data_cleaning.ipynb` | Data cleaning and preprocessing |
| `3_feature_engineering.ipynb` | Construction of behavioral, temporal, and aggregate features |
| `4_feature_selection.ipynb` | KS filter and sequential feature selection |
| `5_model_building_eval.ipynb` | Model training, cross-validation, and hyperparameter tuning |
| `6_impact_evaluation.ipynb` | Performance reporting and financial impact analysis |

The trained model is saved as `fraud_model_bundle.joblib`, which includes the fitted XGBoost model and the list of selected features.

## Key Results

| Metric | Training | Testing | OOT |
|---|---|---|---|
| FDR @ 3% | ~80% | ~77% | ~63% |
| AUC | 0.971 | 0.963 | 0.935 |

The FDR measures how much fraud is captured when reviewing the top N% of transactions ranked by risk score. Training and testing performance are closely aligned, indicating the model generalizes well with minimal overfitting. The OOT set — covering the final two months, unseen during training — shows an expected but moderate performance dip due to temporal distribution shift, while still maintaining strong discriminatory power well above the 0.500 random baseline.

**Recommended cutoff: 5% detection rate — projected net savings of $54.2M annually.**

At this threshold the model captures the majority of fraud while keeping the volume of false positives operationally manageable.

## Notes

- This is a passion project that I'm most proud of to date. Please enjoy. :)
- All analysis is conducted at the transaction level using one year of card transaction data.
- The out-of-time (OOT) validation set covers the final two months of data and is held out entirely from model training and selection.
- See the report for a full discussion of methodology, limitations, and deployment considerations.
