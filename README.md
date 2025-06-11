# iFood Offer Conversion Prediction

This repository presents a complete data science workflow developed to analyze customer behavior and predict offer completion using real-world transactional data from iFood. The project includes exploratory data analysis, feature engineering, model training, and evaluation, with the goal of optimizing offer targeting strategies.

## Table of Contents

- [Project Structure](#project-structure)  
- [Objective](#objective)  
- [Data Overview](#data-overview)  
- [Notebooks](#notebooks)  
- [Key Insights](#key-insights)  
- [Modeling Approach](#modeling-approach)  
- [How to Run](#how-to-run)  
- [Requirements](#requirements)  

---

## Objective

The goal of this project is to explore customer profiles, offer interactions, and transaction patterns to build a machine learning model capable of predicting whether a given offer will be completed. This enables more effective personalization and targeting strategies in marketing campaigns.

---

## Data Overview

The project uses three main data sources:

- `offers.json`: Contains offer attributes like `offer_type`, `min_value`, `duration`, and `discount_value`.
- `profile.json`: Includes customer information such as `age`, `gender`, `credit_card_limit`, and registration date.
- `transactions.json`: Logs customer events, including offer interactions and monetary transactions.

---

## Notebooks

### `1_DataProcessing.py`

This notebook covers the complete data ingestion and preparation process:
- Reads and cleans the `offers`, `profile`, and `transactions` datasets
- Performs feature engineering including user engagement metrics and conversion rates
- Unifies all data into a final modeling dataset

### `2_modeling.py`

This notebook performs the modeling and evaluation:
- Defines the target variable (`is_offer_completed`)
- Conducts correlation analysis to guide feature selection
- Trains a baseline CatBoost model and evaluates its performance
- Applies Optuna for hyperparameter tuning
- Retrains the final model and plots metrics (ROC-AUC, feature importance, confusion matrix)

---

## Key Insights

- Over 58% of received offers were completed, indicating strong engagement.
- Customers who complete offers spend nearly 3x more than those who do not.
- `reward` and `discount_value` are redundant and should not be used together.
- Age and credit card limit show moderate positive correlation (r ≈ 0.41).
- No significant linear correlation found between account age and engagement.

---

## Modeling Approach

The final model is a CatBoostClassifier optimized using Optuna. Performance is measured using ROC-AUC, precision, recall, and other relevant metrics. Feature importance plots and optimal threshold analysis are included to support decision-making.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ifood-offer-prediction.git
   cd ifood-offer-prediction
