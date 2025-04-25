# Real-Time Credit Card Fraud Detection

## Project Description

This project implements a real-time fraud detection system for credit card transactions using machine learning.  
Users can upload transaction data and receive immediate fraud predictions through a web-based interface.  
If the uploaded data includes true labels, the system will evaluate prediction accuracy and display metrics.

---

## Data Source

Kaggle - Credit Card Fraud Detection (2023)  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- Features: `V1` to `V28` (anonymized)
- `Amount`: transaction amount
- `Class`: 0 = normal, 1 = fraud  
- The dataset is artificially balanced for equal class distribution.

---

## Required Packages

Install all required packages with:

```
pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
```

Dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `joblib`
- `matplotlib`
- `seaborn`

---

## How to Run

1. **Train and save the model:**

```
python model/model_utils.py
```

2. **Launch the Streamlit web app:**

```
streamlit run app/streamlit_app.py
```

3. **Use the Web Interface:**

- Upload a `.csv` file with columns: `V1` to `V28`, and `Amount`
- (Optional) Include the `Class` column for accuracy evaluation
- View fraud predictions and download labeled results

---
