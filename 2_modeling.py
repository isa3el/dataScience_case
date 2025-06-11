# Databricks notebook source
# MAGIC %md
# MAGIC # iFood Data Science Case: Offer Conversion Prediction
# MAGIC
# MAGIC This notebook walks through a complete end-to-end workflow to predict whether an iFood offer will be completed by a customer:
# MAGIC
# MAGIC 1. **Data Ingestion & Exploration**  
# MAGIC    - Read the processed dataset from Hive  
# MAGIC    - Display basic summary and distribution of key variables  
# MAGIC
# MAGIC 2. **Target definition**  
# MAGIC    - Create a binary target column (`is_offer_completed`) indicating if an offer was completed  
# MAGIC
# MAGIC 3. **Correlation Analysis**  
# MAGIC    - Compute and visualize the Spearman correlation matrix to understand relationships among features  
# MAGIC
# MAGIC 4. **Feature Preparation**  
# MAGIC    - Select relevant numeric and categorical features  
# MAGIC    - Handle missing values and split into training and test sets  
# MAGIC
# MAGIC 5. **Baseline Modeling with CatBoost**  
# MAGIC    - Train a default CatBoostClassifier  
# MAGIC    - Evaluate performance via ROC-AUC, classification report and confusion matrix  
# MAGIC
# MAGIC 6. **Hyperparameter Tuning (Optuna)**  
# MAGIC    - Define an Optuna study to optimize CatBoost hyperparameters  
# MAGIC    - Run trials to maximize ROC-AUC on the hold-out set  
# MAGIC
# MAGIC 7. **Final Model Training & Evaluation**  
# MAGIC    - Retrain the optimal CatBoost model with best parameters  
# MAGIC    - Assess final performance (ROC-AUC, precision/recall, optimal threshold)  
# MAGIC    - Plot feature importances and final confusion matrix and ROC curve 
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %pip install catboost
# MAGIC %pip install optuna

# COMMAND ----------

# Standard library
from functools import reduce

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    MultilayerPerceptronClassifier,
    RandomForestClassifier as SparkRFClassifier
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col, when

# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# CatBoost
from catboost import CatBoostClassifier, Pool

# Hyperparameter optimization
import optuna

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. Data Ingestion & Exploration

# COMMAND ----------

# Reading processed data
dataset = spark.table("train_df_json")


# COMMAND ----------

dataset.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Target definition

# COMMAND ----------

# Create a new binary column 'is_offer_completed' to be target variable
dataset = dataset.withColumn(
    "is_offer_completed",
    when(col("n_offers_completed") > 0, 1).otherwise(0)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Correlation Analysis

# COMMAND ----------

dataset_pd = dataset.toPandas()

# Compute the Spearman correlation matrix between all numeric columns
corr = dataset_pd.corr(method='spearman')

plt.figure(figsize=(20, 10))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",  # Color map: red for positive, blue for negative
    center=0,
    linewidths=.5,
    cbar_kws={'label': 'Correlação'}
)

plt.title("Correlação (spearman)", fontsize=16, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Feature Preparation

# COMMAND ----------


# Define target column
target = "is_offer_completed"


#  Define the list of input features to be used by the model.
features = [
    "n_transactions", "n_offers_received", "n_offers_viewed", "avg_amount", "spending_ratio", "age", "credit_card_limit", "min_value", "discount_value", "duration", "offer_id"
]


# Split the dataset into training and testing sets (80% train, 20% test) with a fixed seed for reproducibility
train_df, test_df = dataset.randomSplit([0.8, 0.2], seed=42)

X_train = train_df.select(features).toPandas()
X_test = test_df.select(features).toPandas()


# Extract the target values for training and test flatten the array
y_train = train_df.select(target).toPandas().values.ravel()
y_test = test_df.select(target).toPandas().values.ravel()

# Preparing full dataset for final run
full_df = dataset.select(features).toPandas()
full_target_df = dataset.select(target).toPandas().values.ravel()


# COMMAND ----------

# Identify the categorical columns based on data types (object or category)
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Fill missing values in numeric columns with -999
# CatBoost can handle missing values, but filling is more consistent
X_train.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)
full_df.fillna(-999, inplace=True)



# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Baseline Modeling with CatBoost

# COMMAND ----------

# Initialize the CatBoostClassifier
model = CatBoostClassifier(
    iterations=200,         # Number of boosting rounds (trees)
    learning_rate=0.1,      # Step size shrinkage used to prevent overfitting
    depth=4,                # Maximum depth of each tree
    eval_metric='AUC',      # Evaluation metric: Area Under the ROC Curve
    verbose=100             # Print training progress every 100 iterations
)

# Train the model on the training set
# 'cat_features' specifies which columns are categorical
# 'eval_set' is used to monitor performance on the test set during training
# 'plot=True' shows a live plot of the learning curve (requires Jupyter environment)
model.fit(X_train, y_train, cat_features=categorical_cols, eval_set=(X_test, y_test), plot=True)


# COMMAND ----------

# Generate class predictions on the test set
y_pred = model.predict(X_test)

# Generate predicted probabilities for the positive class (class “1”)
# predict_proba returns an array [P(class=0), P(class=1)] for each sample
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print a detailed classification report (precision, recall, f1-score, support)
print(classification_report(y_test, y_pred))

# 5. Compute and print the ROC AUC score, which measures how well the model separates the two classes across all possible thresholds
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))


# COMMAND ----------

# Plot confusion matrix from true labels (y_test) and predictions (y_pred)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predict")
plt.ylabel("True")
plt.title("Confusion Matrix")

plt.show()

# True Negatives: 5,258 cases where the model correctly did not recommend the offer to customers who did not convert.
# False Positives: 1,366 times it suggested the offer to non-converters (“bothering” uninterested customers).
# False Negatives: 434 missed converters (the model did not recommend to customers who would have bought).
# True Positives: 47,301 cases where it correctly recommended the offer to customers who converted.

# COMMAND ----------

# Using Optuna hyperparameter optimization for the CatBoostClassifier,
# using ROC AUC on the held-out test set as the objective to maximize

def objective(trial):
    params = {
        "iterations":      trial.suggest_int("iterations",  200, 500),
        "depth":           trial.suggest_int("depth",       3,    6),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg",   1,    10),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "border_count":    trial.suggest_int("border_count",  32,   255),
        "eval_metric":     "AUC",        
        "loss_function":   "Logloss",    
        "verbose":         0             
    }

    #Initialize a CatBoostClassifier with the sampled hyperparameters
    model = CatBoostClassifier(**params)

    # Train on the training set, validating on the test set
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        cat_features=categorical_cols
    )

    # Predict probabilities of the positive class on the test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute and return the ROC AUC score (the objective to maximize)
    return roc_auc_score(y_test, y_pred_proba)

# Create an Optuna study configured to maximize the objective
study = optuna.create_study(direction="maximize")

# Run the optimization for a fixed number of trials. 
# # Currently using 2 for a quick test; for a full run, set bigger n_trials.
study.optimize(objective, n_trials=2)

# Print out the best AUC and corresponding hyperparameters
print("Best AUC:", study.best_value)
print("Best hyperparameters:")
for name, value in study.best_params.items():
    print(f"  {name}: {value}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### 6. Hyperparameter Tuning (Optuna)

# COMMAND ----------

# get feature importance scores from the trained model
importances = model.get_feature_importance()
feat_imp_df = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp_df.sort_values(by='importance', ascending=False).head(10)


# COMMAND ----------

# MAGIC %md
# MAGIC ### 7. Final Model Training & Evaluation

# COMMAND ----------

# Extract the best hyperparameters found by Optuna
best_params = study.best_params

final_model = CatBoostClassifier(
    **best_params,
    cat_features=categorical_cols,
    verbose=100
)

# Train the final model on full data
final_model.fit(full_df, full_target_df, eval_set=(full_df, full_target_df))

# COMMAND ----------

full_pred_proba = final_model.predict_proba(full_df)[:,1]
full_df["score"] = full_pred_proba
print("ROC AUC:", roc_auc_score(full_target_df, full_pred_proba))

# classificação usando threshold 0.9 
full_pred = (full_pred_proba >= 0.9).astype(int)
print(classification_report(full_target_df, full_pred))

# COMMAND ----------


# Plotting and ranking feature importance
fi = pd.Series(final_model.get_feature_importance(), index=X_train.columns)
fi_sorted = fi.sort_values()

colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(fi_sorted)))
fig, ax = plt.subplots(figsize=(6,8))
ax.barh(fi_sorted.index, fi_sorted.values, color=colors)
ax.set_xlabel("Importância")
ax.set_title("Importância das Features")
plt.tight_layout()
plt.show()


# COMMAND ----------


# Compute confusion matrix: rows=true labels, columns=predicted labels
cm = confusion_matrix(y_test, y_pred, labels=[0,1])

# Plot confusion matrix with a red gradient
fig, ax = plt.subplots(figsize=(6,6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(ax=ax, cmap='Reds', colorbar=False)  # set colorbar=True to show the legend
ax.set_title("Confusion Matrix (cmap=Reds)")
plt.tight_layout()
plt.show()


# COMMAND ----------


fpr, tpr, thresholds = roc_curve(full_target_df, full_pred_proba)

# Calculate the AUC value
roc_auc = auc(fpr, tpr)

# Find the index of the optimal threshold (maximizes TPR – FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

# Compute percentage of cases captured at the optimal point
percent_aproveitadas = optimal_tpr * 100

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve', color='black')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random-chance diagonal

# Highlight the optimal point
plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='darkred', label='Optimal Point')
plt.axvline(x=optimal_fpr, linestyle='--', color='darkred')
plt.axhline(y=optimal_tpr, linestyle='--', color='darkred')

plt.title("ROC Curve AUC", fontsize=14, weight='bold')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')

# Output the percentage of opportunities seized
print(f"Percent of opportunities seized at optimal threshold: {percent_aproveitadas:.2f}%")

plt.show()


# COMMAND ----------

dataset_pd = dataset.toPandas()
full_df["account_id"] = dataset_pd["account_id"]
full_df.head()

# COMMAND ----------

evaluated_pairs = (
    dataset
      .select("account_id", "offer_id")   # pick the two columns
      .distinct()                         # drop duplicates
      .count()                            # count the resulting rows
)

# 2) How many of those exceed our cutoff (i.e. would be recommended)?
threeshoulder = 0.4
recommended_pairs = (
                     full_df[full_df["score"] > threeshoulder]
                    .drop_duplicates(["account_id", "offer_id"])
                    .shape[0]
                    )

users_reached = full_df.loc[full_df["score"] > threeshoulder, "account_id"].nunique()

total_users           = full_df["account_id"].nunique()
total_offers          = full_df["offer_id"].nunique()
total_possible_pairs  = total_users * total_offers

total_possible_pairs    = total_users * total_offers
evaluated_pairs     = evaluated_pairs    / total_possible_pairs
users_reached        = users_reached      / total_users
opportunity_captured = recommended_pairs  / total_possible_pairs

# Print key opportunity metrics at the chosen cutoff
print(f"Total combinations possibilities:       {total_possible_pairs}")
print(f"Pairs evaluated out of all possible:       {evaluated_pairs:.2%}")
print(f"Share of users reached with ≥T recommendation: {users_reached:.2%}")
print(f"Overall opportunity captured:               {opportunity_captured:.2%}")


# COMMAND ----------


