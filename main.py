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
    # Parameters (set here easily)
    mode = "train"  # options: "train" or "predict"
    use_smote = True
    use_gridsearch = True
    use_random_forest = False
    variance_threshold = 0.01
    threshold_for_prediction = 0.5  # Used only for evaluation or prediction

    data_path = os.path.join("data", "creditcard_2023.csv")

    if mode == "train":
        print("Running training mode...\n")

        # Load and preprocess data
        X, y = aux_1.load_data(data_path)
        X_scaled, scaler = aux_1.preprocess_data(X, variance_threshold=variance_threshold)
        X_train, X_test, y_train, y_test = aux_1.split_data(X_scaled, y)

        # Optionally apply SMOTE
        if use_smote:
            X_train, y_train = aux_1.apply_smote(X_train, y_train)

        # Choose and train model
        if use_random_forest:
            model = aux_1.train_random_forest(X_train, y_train)
        else:
            model = aux_1.train_logistic_model(X_train, y_train, use_gridsearch=use_gridsearch)

        # Evaluate model
        aux_1.evaluate_model(model, X_test, y_test, threshold=threshold_for_prediction)

        # Save model and scaler
        aux_1.save_model(model, scaler)

    elif mode == "predict":
        print("Running prediction mode...\n")

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

        # If model supports predict_proba, allow threshold adjustment
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:, 1]
            predictions = (probs >= threshold_for_prediction).astype(int)
        else:
            predictions = model.predict(X_scaled)

        df['Prediction'] = predictions
        df['Prediction_Label'] = df['Prediction'].map({0: "Normal", 1: "Fraud"})

        output_path = "predictions.csv"
        df.to_csv(output_path, index=False)
        print(f"Prediction complete. Results saved to {output_path}")

    else:
        print("Error: Unknown mode. Set mode = 'train' or 'predict'.")

if __name__ == "__main__":
    main()
