# To run the Streamlit web application locally, follow these steps:
# Open a terminal and navigate to the project directory.
# Ensure that your Python virtual environment is activated.
# Run the following command:
# streamlit run app/streamlit_app.py

import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Add model folder to sys.path (safe for deployment)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

# Direct import of aux_1
import aux_1

# Load trained model and scaler
@st.cache_resource
def load_model():
    model, scaler = aux_1.load_model_and_scaler()
    return model, scaler

# UI setup
st.set_page_config(page_title="Credit Card Fraud Detector", layout="wide")
st.title("Real-Time Credit Card Fraud Detection")

st.markdown("""
Upload transaction data and receive real-time fraud predictions.  
If your data includes the `Class` column, the app will evaluate prediction accuracy and show the confusion matrix.
""")

uploaded_file = st.file_uploader("Upload CSV file (must contain V1–V28 and Amount)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())

        required_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
        if not all(col in df.columns for col in required_cols):
            st.error("Uploaded CSV is missing required columns (V1–V28 + Amount).")
        else:
            model, scaler = load_model()

            X_input = df[required_cols]
            X_scaled = scaler.transform(X_input)
            preds = model.predict(X_scaled)
            df['Prediction'] = preds
            df['Prediction_Label'] = df['Prediction'].map({0: "Normal", 1: "Fraud"})

            st.subheader("Prediction Summary")
            st.bar_chart(df['Prediction_Label'].value_counts())

            if 'Class' in df.columns:
                y_true = df['Class']
                y_pred = df['Prediction']
                acc = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                st.subheader("Evaluation on Labeled Data")
                st.write(f"Accuracy: {acc:.2%}")

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                            xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            st.subheader("Prediction Results")
            st.dataframe(df[['Prediction_Label']])

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv_download,
                               file_name="fraud_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
