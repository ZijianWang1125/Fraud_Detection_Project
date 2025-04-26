import pandas as pd
import os
import sys
import importlib.util

# Dynamically load aux_1 from model/aux_1.py
model_utils_path = os.path.join("model", "aux_1.py")
spec = importlib.util.spec_from_file_location("aux_1", model_utils_path)
aux_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aux_1)

def main():
    data_path = os.path.join("data", "creditcard_2023.csv")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at: {data_path}")
        return

    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    try:
        model, scaler = aux_1.load_model_and_scaler()
    except Exception as e:
        print(f"Error: Failed to load model or scaler: {e}")
        return

    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)
    df['Prediction'] = predictions
    df['Prediction_Label'] = df['Prediction'].map({0: "Normal", 1: "Fraud"})

    output_path = "predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Prediction complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
