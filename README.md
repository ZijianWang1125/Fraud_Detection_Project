# Real-Time Credit Card Fraud Detection

This project builds a real-time fraud detection system using machine learning techniques.  
It includes model training, evaluation, deployment of a Streamlit web application, and full prediction functionality.

---

## Project Description

- Developed a baseline fraud detection model using Logistic Regression and Random Forest.
- Applied feature preprocessing, variance thresholding, and SMOTE oversampling.
- Conducted hyperparameter tuning via GridSearchCV to optimize model performance.
- Deployed an interactive Streamlit application for real-time fraud prediction based on uploaded CSV files.

---

## Dataset Description

- **Source:** Kaggle Credit Card Fraud Detection dataset (2023 version).
- **Records:** 568,630 transactions.
- **Features:** V1–V28 (anonymized PCA components) and Amount.
- **Target:** `Class` (0 = Normal, 1 = Fraud).

The dataset used here is artificially balanced for initial training purposes.

---

## Required Packages

- pandas
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- joblib
- streamlit

Install them using:

```bash
pip install -r requirements.txt
```

---

## How to Run the Code

### 1. Training the Model

```bash
python main.py
```
- In `main.py`, set:
  - `mode = "train"` for training
  - Configure options like `use_smote`, `use_gridsearch`, and `use_random_forest` as needed.

The trained model (`fraud_model.pkl`) and scaler (`scaler.pkl`) will be saved in the `model/` directory.

---

### 2. Predicting New Data

```bash
python main.py
```
- In `main.py`, set:
  - `mode = "predict"` for making predictions.

Predicted results will be saved as `predictions.csv`.

---

### 3. Running the Streamlit Web Application

```bash
streamlit run app/streamlit_app.py
```
- Upload a CSV file containing V1–V28 and Amount columns.
- Instantly receive fraud prediction results.
- Download prediction output if needed.

---

## Current Improvements

- Added variance-based feature filtering.
- Applied SMOTE to address class imbalance.
- Performed hyperparameter tuning for Logistic Regression.
- Supported Random Forest as an alternative model.
- Enabled customizable classification thresholds.

---

## Future Work

- Explore adding time-based and behavioral features.
- Implement online learning for continuous fraud detection updates.
- Expand evaluation on real-world imbalanced datasets.

---
