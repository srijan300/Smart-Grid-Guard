import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import os

from model_utils import lstm_forecast, train_lstm_model
from anomaly_utils import detect_anomalies

# -----------------------
# Config
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.keras")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")
DATA_PATH = os.path.join(BASE_DIR, "elf_dataset.xlsx")
TARGET_COL = "DEMAND"

# -----------------------
# Load model + scalers
# -----------------------
@st.cache_resource
def load_model_and_scalers():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        return model, scaler_X, scaler_y
    else:
        return None, None, None


# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)


# -----------------------
# Streamlit App
# -----------------------
st.set_page_config(
    page_title="SmartGridGuard | Energy Forecasting & Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ SmartGridGuard")
st.markdown("### 🧠 AI-Powered Short-Term Energy Load Forecasting & Contextual Anomaly Detection")

# File uploader for retraining
with st.sidebar:
    st.header("⚙️ Configuration & Retraining")
    uploaded_file = st.file_uploader("Upload new dataset (.xlsx) to retrain model", type=["xlsx"])
    score_threshold = st.slider("Anomaly Percentile Threshold", min_value=90.0, max_value=99.9, value=98.5, step=0.1)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Retrain model on new dataset
    st.info("🔄 Training new LSTM model with uploaded dataset...")
    model, scaler_X, scaler_y, history, results = train_lstm_model(df, TARGET_COL, save_dir=BASE_DIR)

    # Save artifacts
    model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    st.success("✅ Model retrained and saved successfully!")
else:
    df = load_data()

# Show dataset preview
with st.expander("📂 Dataset Preview", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# Load pretrained model + scalers
model, scaler_X, scaler_y = load_model_and_scalers()
if model is None:
    st.error("No pre-trained model found. Please upload dataset to train.")
    st.stop()

# -----------------------
# Forecasting & Evaluation
# -----------------------
with st.spinner("Generating forecasts and detecting anomalies..."):
    lstm_preds, y_true_lstm = lstm_forecast(model, scaler_X, scaler_y, df, TARGET_COL)

    # Detect anomalies using score percentile
    anomalies, iso_model = detect_anomalies(
        df, 
        y_true_lstm, 
        lstm_preds, 
        contamination=None,
        score_percentile=score_threshold
    )

# Compute metrics
from model_utils import evaluate_model
eval_metrics = evaluate_model(y_true_lstm, lstm_preds)

st.subheader("📊 Key Performance Metrics")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("RMSE", f"{eval_metrics['RMSE']:.2f}")
m_col2.metric("MAE", f"{eval_metrics['MAE']:.2f}")
m_col3.metric("R² Score", f"{eval_metrics['R²']:.4f}")
m_col4.metric("Flagged Anomalies", f"{len(anomalies)} ({len(anomalies)/len(lstm_preds)*100:.1f}%)")

# -----------------------
# Plots
# -----------------------
st.subheader("📈 Real-Time Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**LSTM Forecast vs Actual Load**")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_true_lstm, label="Actual Demand", color="#1f77b4", linewidth=1.2)
    ax.plot(lstm_preds, label="LSTM Forecast", color="#ff7f0e", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time Step (Hours)")
    ax.set_ylabel("Demand (MW)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    st.pyplot(fig)

with col2:
    st.markdown("**Contextual Anomaly Detection (Isolation Forest)**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(df[TARGET_COL].values, label="Actual Demand", color="#1f77b4", alpha=0.7, linewidth=1.2)
    ax2.scatter(
        anomalies["index"], 
        anomalies["actual"], 
        color="#d62728", 
        label="Detected Anomaly", 
        s=28, 
        zorder=5
    )
    ax2.set_xlabel("Time Step (Hours)")
    ax2.set_ylabel("Demand (MW)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    st.pyplot(fig2)

# Anomalies Table
if not anomalies.empty:
    with st.expander(f"🚨 View Flagged Anomalies ({len(anomalies)} events)", expanded=False):
        st.dataframe(
            anomalies[["index", "actual", "predicted", "abs_error", "score"]].rename(
                columns={
                    "index": "Time Index",
                    "actual": "Actual Demand",
                    "predicted": "Forecast Demand",
                    "abs_error": "Absolute Error",
                    "score": "Anomaly Score"
                }
            ).head(50),
            use_container_width=True
        )
