import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Always use absolute path based on current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def split_data(X_scaled, y, test_size=0.2, random_state=42):
    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")

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
    data_path = os.path.join(BASE_DIR, "..", "data", "creditcard_2023.csv")
    X, y = load_data(data_path)
    X_scaled, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, scaler)
    
