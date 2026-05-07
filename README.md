# Lending Club Loan Default Prediction

**Module:** Machine Learning — Group Project  
**Topic:** Explainable Credit Risk Modelling for P2P Lending (Lending Club)

---

## Overview

This project builds and compares several machine learning models to predict **loan default** on a filtered sample of the **Lending Club** peer-to-peer (P2P) lending dataset. The focus is on balancing **predictive performance** with **regulatory transparency**, aligned with the **EU AI Act** requirement that credit scoring is a high‑risk AI application that must remain explainable and auditable.

A class‑weighted logistic regression model serves as a transparent glass‑box baseline and is compared against more complex models, including Random Forest, XGBoost, LightGBM and a simple neural network. Performance is evaluated on an imbalanced default vs. non‑default problem using ROC‑AUC, precision, recall and confusion matrices.

---

## Objectives

- Build an **explainable credit risk modelling framework** for P2P lending using the Lending Club dataset
- Engineer a binary target that distinguishes **bad loans** (defaults and late payments) from **good loans** (fully paid)
- Preprocess mixed numerical and categorical borrower features through a single, reproducible pipeline
- Train and evaluate:
  - Class‑weighted **Logistic Regression** (glass‑box baseline)
  - **Random Forest**
  - **XGBoost**
  - **LightGBM**
  - **Neural Network** (MLP)
- Compare performance on the minority default class and understand trade‑offs between:
  - Predictive power
  - Financial error cost (false positives vs false negatives)
  - Model transparency for regulators and investors

---

## Dataset

- Source: Lending Club consumer loan data (filtered sample)
- Target variable: **binary loan status** (`target_default`) where:
  - `1` = bad loan (default / late)
  - `0` = good loan (fully paid)
- Features include:
  - Loan‑level: `loan_amnt`, `funded_amnt`, `term`, `int_rate`, `installment`
  - Borrower‑level: `annual_inc`, `dti`, `home_ownership`, `purpose`
  - Credit history: `fico_range_low`, `fico_range_high`, `open_acc`, `revol_bal`, `revol_util`, `total_acc`
- Class imbalance:
  - Around **40,563** non‑default vs **9,437** default cases in the working sample

---

## Methods & Models

### Data preprocessing

- Sampling strategy: ~80% of overall data used with a fixed random state for reproducibility
- Target engineered to 0/1 (good vs bad loans)
- Preprocessing pipeline implemented as a **scikit‑learn `ColumnTransformer`**, applied consistently to train and test splits:
  - Numerical features: median imputation + `StandardScaler`
  - Categorical features: one‑hot encoding

### Baseline: Logistic Regression

- Class‑weighted logistic regression as the **glass‑box** baseline
- Balanced class weights to reflect higher cost of misclassifying a default loan than a good loan
- Up to 1000 iterations to allow the solver to converge on stable coefficients

### Tree‑based ensembles: Random Forest and XGBoost

- **Random Forest** with ~200 trees and balanced subsampling to retain sensitivity to the default class
- **XGBoost (Extreme Gradient Boosting)** to test whether boosting provides better ranking of default risk than bagging in structured credit data

### Advanced models: LightGBM and Neural Network

- **LightGBM** as a faster, efficient boosting alternative for large, sparse design matrices
- **Neural Network (MLP)** with ReLU activations as a deep learning benchmark, operating at the black‑box end of the spectrum

---

## Evaluation

- Metrics:
  - **ROC‑AUC** as the main ranking metric under class imbalance
  - Classification reports for **precision**, **recall** and **F1** on the default class
  - Confusion matrices to understand the financial trade‑offs of false positives vs false negatives
- Comparative findings (high level):
  - XGBoost and LightGBM achieve the **highest ROC‑AUC**, followed by Random Forest
  - Logistic regression baseline has lower ROC‑AUC but remains robust and highly interpretable
  - Neural network is competitive but does not clearly outperform gradient boosting on this structured dataset

---

## Files in this repository

- `ML-Lending-Club.ipynb` — end‑to‑end modelling notebook (data exploration, preprocessing, model training and evaluation)
- `ML-Lending-Club.pdf` — project report: *“Explainable Credit Risk Modelling for P2P Lending: Interpreting Loan Default under EU AI Act”*

---

## Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3** | Core programming language |
| **pandas / NumPy** | Data wrangling and numerical operations |
| **scikit‑learn** | Pipelines, preprocessing, logistic regression, metrics |
| **RandomForestClassifier** | Ensemble model for non‑linear relationships |
| **XGBoost** | Gradient boosting for structured credit risk data |
| **LightGBM** | Efficient boosting on large, sparse feature sets |
| **MLPClassifier** | Neural network baseline |
| **Matplotlib / Seaborn** | Visualisation of distributions and model performance |
| **Jupyter Notebook** | Interactive experimentation and documentation |

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/nizamshaikh12/lending-club-loan-default-prediction.git
   ```
2. Open `ML-Lending-Club.ipynb` in **Jupyter Notebook**, **VS Code**, or **Google Colab**
3. Ensure the Lending Club dataset CSV is available at the path expected in the notebook
4. Run the notebook cells in order to:
   - Explore the data
   - Build the preprocessing pipeline
   - Train and evaluate all five models
   - Review confusion matrices and ROC curves

> **Requirements:** Python 3.x, pandas, numpy, scikit‑learn, xgboost, lightgbm, matplotlib, seaborn, jupyter

---

## Acknowledgements

This was a **group project**. The work includes joint modelling and reporting; this repository hosts the shared notebook and final report for portfolio and learning purposes.
