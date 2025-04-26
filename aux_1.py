import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import SMOTE

# Always use absolute path based on current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Data loading
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

# Feature Preprocessing
def preprocess_data(X, variance_threshold=None):
    # Optional variance threshold filtering
    if variance_threshold is not None:
        selector = VarianceThreshold(threshold=variance_threshold)
        X = selector.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# Data Splitting
def split_data(X_scaled, y, test_size=0.2, random_state=42):
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

# SMOTE Over-sampling
def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Logistic Regression Training
def train_logistic_model(X_train, y_train, use_gridsearch=False):
    if use_gridsearch:
        param_grid = {'C': [0.01, 0.1, 1, 10]}
        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
    return model

# Random Forest Training
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Model Evaluation
def evaluate_model(model, X_test, y_test, threshold=0.5):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

# Model Saving
def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

# Model Loading
def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        raise FileNotFoundError("Saved model or scaler not found.")

# Local training entry point
if __name__ == "__main__":
    print("Running local training and evaluation...\n")
    # Load data
    X, y = load_data("../data/creditcard_2023.csv")

    # Preprocess with variance filter (optional, set None if not needed)
    X_scaled, scaler = preprocess_data(X, variance_threshold=0.01)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Optionally apply SMOTE
    X_train, y_train = apply_smote(X_train, y_train)

    # Train model (choose logistic regression or random forest)
    model = train_logistic_model(X_train, y_train, use_gridsearch=True)
    # model = train_random_forest(X_train, y_train)  # Uncomment if want random forest

    # Evaluate model
    evaluate_model(model, X_test, y_test, threshold=0.5)

    # Save model and scaler
    save_model(model, scaler)
