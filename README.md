# ⚡ SmartGridGuard

### 🧠 AI-Powered Short-Term Energy Load Forecasting & Contextual Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 👥 Authors & Internship Project Contributors

* **Srijan Paul** ([@srijan300](https://github.com/srijan300))
* **Sristi Saha**
* **Sahana Samanta**

> **Internship Experience & Group Project:** Developed as an end-to-end Machine Learning and Smart Grid analytics project focused on sustainable energy management, operational efficiency, and fault resilience for modern power grids.

---

## 🌍 Inspiration & Problem Statement

Buildings and industrial facilities account for nearly **one-third of global energy consumption and greenhouse gas emissions**. In modern smart grids and decentralized energy systems, **accurate Short-Term Load Forecasting (STLF)** is vital for:
* Balancing electricity demand with generation in real time.
* Seamlessly integrating intermittent renewable energy sources (solar, wind).
* Mitigating costly peak-demand surcharges and grid blackouts.
* Early detection of operational faults, power leakages, and equipment anomalies.

Traditional time-series models often struggle with complex non-linear temporal dependencies and frequently trigger false alarms during legitimate holiday or weekend consumption shifts. 

To overcome these challenges, we built **SmartGridGuard** — a hybrid deep learning and contextual anomaly detection platform.

---

## ⚙️ What It Does

**SmartGridGuard** combines deep recurrent neural networks with unsupervised machine learning to deliver proactive grid monitoring:

1. **Deep Learning Forecasting:** Implements a stacked **LSTM (Long Short-Term Memory)** network trained on multi-step sequence windows to forecast short-term electricity load.
2. **Contextual Anomaly Detection:** Applies an **Isolation Forest** on both forecast residuals (actual vs. predicted errors and rolling baseline shifts) and calendar context (cyclical day-of-week encoding, weekends, and holidays).
3. **Adaptive Thresholding:** Uses dynamic score percentile cutoffs rather than fixed contamination rates to filter noise and isolate true consumption spikes and drops.
4. **Interactive Operations Dashboard:** A production-ready **Streamlit interface** featuring live load curves, interactive anomaly inspection tables, and dynamic on-the-fly model retraining with new data uploads.

```
Predict (LSTM) ➔ Compare (Residuals) ➔ Contextualize (Calendar) ➔ Detect (Isolation Forest) ➔ Visualize (Streamlit)
```

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Historical Energy Data / Kaggle ELF Dataset] --> B[Data Preprocessing & Cleaning]
    B --> C[Feature Engineering: Cyclical DOW, Weekend, Holiday]
    B --> D[MinMax Feature & Target Normalization]
    D --> E[Sliding Window Sequence Generator (t=20)]
    
    E --> F[Stacked LSTM Architecture (64 -> Dropout 0.2 -> 32 -> Dense 1)]
    F --> G[Short-Term Load Forecast]
    
    G --> H[Residual Calculation: Error & Rolling Baselines]
    C --> I[Contextual Feature Matrix]
    H --> I
    
    I --> J[StandardScaler Normalization]
    J --> K[Isolation Forest Unsupervised Detection]
    K --> L[Percentile-Based Score Thresholding]
    
    G --> M[Streamlit Interactive Dashboard]
    L --> M
    M --> N[Real-Time Visualizations & Actionable Alerts]
```

---

## 🔬 Methodology & Core Components

### 1. Data Engineering & Preprocessing
* **Dataset:** Electricity Load Forecasting (ELF) Dataset containing hourly power consumption records alongside weather and lag features (`DEMAND`, `T2M_toc`, `week_X-2`, `week_X-3`, `week_X-4`, `MA_X-4`, `dayOfWeek`, `weekend`, `holiday`, `hourOfDay`).
* **Sequence Windowing:** Formulated lookback sequences ($T = 20$ time steps) to capture short-term diurnal patterns and temporal dependencies.
* **Feature Scaling:** Independent `MinMaxScaler` pipelines for multivariate exogenous inputs ($X$) and univariate target load ($y$) to avoid lookahead bias and maintain scaling integrity.

### 2. Deep Forecasting Network (LSTM)
* **Architecture:**
  * Input Sequence Layer: $(20, N_{features})$
  * LSTM Layer 1: 64 hidden units, `return_sequences=True`
  * Regularization: Dropout ($0.20$) to prevent overfitting
  * LSTM Layer 2: 32 hidden units
  * Regularization: Dropout ($0.20$)
  * Output Layer: Dense(1) with linear activation
* **Optimization:** Adam optimizer with Mean Squared Error (MSE) loss function.

### 3. Contextual Anomaly Detection
* **Residual & Baseline Modeling:**
  $$\text{Residual}_t = (Y_t - \mu_{\text{rolling}, t}) - (\hat{Y}_t - \hat{\mu}_{\text{rolling}, t})$$
* **Contextual Features:**
  * Sine/Cosine cyclical encodings: $\sin(2\pi \cdot \text{DOW} / 7)$ and $\cos(2\pi \cdot \text{DOW} / 7)$
  * Binary flags for weekends and recognized holidays
* **Isolation Forest Engine:** 300 estimators fit on scaled residual and contextual features.
* **Percentile Thresholding:** Dynamically isolates extreme deviation percentiles (e.g. top 1.5% anomalies), reducing false alarms caused by holiday drops or scheduled operational surges.

### 4. Interactive Streamlit Dashboard
* Real-time plotting of actual demand versus model predictions.
* Scatter overlay marking pinpointed anomalies along the time horizon.
* Metric summary cards for instant evaluation (RMSE, MAE, R², Anomaly Count).
* In-app file uploader allowing operations teams to upload `.xlsx` log files to trigger automated pipeline retraining and model persistence.

---

## 📂 Repository Structure

```
Smart-Grid-Guard/
├── app.py                # Streamlit web application & visualization dashboard
├── model_utils.py        # LSTM sequence generation, training, and inference utilities
├── anomaly_utils.py      # Contextual Isolation Forest anomaly detection pipeline
├── data_utils.py         # Data loading and verification utilities
├── train_lstm.py         # Standalone CLI training script
├── elf_dataset.xlsx      # Benchmark Electricity Load Forecasting dataset
├── lstm_model.keras      # Pre-trained deep learning LSTM model weights
├── scaler_X.pkl          # Pickled MinMax feature scaler
├── scaler_y.pkl          # Pickled MinMax target scaler
├── requirements.txt      # Python package dependencies
├── .gitignore            # Git exclusion rules
└── README.md             # Project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.10 or 3.11
* `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/srijan300/Smart-Grid-Guard.git
cd Smart-Grid-Guard
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to explore the dashboard.

### 5. (Optional) Retrain via Command Line
```bash
python train_lstm.py
```

---

## 📊 Evaluation & Performance Metrics

The model is evaluated using standard forecasting and error metrics:

| Metric | Formula | Description |
| :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2}$ | Root Mean Squared Error (penalizes large load variances) |
| **MAE** | $\frac{1}{n}\sum \|y_t - \hat{y}_t\|$ | Mean Absolute Error (average magnitude of forecasting error) |
| **MAPE** | $\frac{1}{n}\sum \left\|\frac{y_t - \hat{y}_t}{y_t}\right\| \times 100\%$ | Mean Absolute Percentage Error |
| **$R^2$ Score** | $1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$ | Coefficient of Determination |

---

## 🧩 Challenges Overcome

1. **False Positives in Anomaly Detection:** Unfiltered anomaly detectors flag natural holidays or weekend load drops as grid failures. Resolved by feeding cyclical day-of-week and holiday binary flags into the Isolation Forest feature matrix.
2. **Time-Series Sequence Alignment:** Fixed sequence-window lag offsets ($t-20$) to guarantee zero data leakage and preserve temporal alignment between actual demand and flagged anomalies.
3. **Adaptive Thresholding:** Replaced rigid contamination percentages with distribution-based percentile ranking, ensuring flexibility across diverse seasonal patterns.
4. **Interactive Dashboard Latency:** Leveraged Streamlit caching (`@st.cache_resource`, `@st.cache_data`) for sub-second load times during interactive exploration.

---

## 🔮 Future Enhancements

* 🌐 **Time-Series Foundation Models (TSFM):** Experiment with pre-trained zero-shot foundation models (e.g., Chronos, TimesFM) for transfer learning across varied geographic zones.
* 🔍 **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) values to diagnose the specific physical drivers behind each flagged anomaly.
* ⚡ **Edge & IoT Streaming:** Ingest continuous telemetry feeds via Apache Kafka and MQTT protocols from industrial smart meters.
* 🤖 **Autonomous Remediation:** Connect anomaly alerts with automated Demand Response (DR) load-shedding and battery energy storage system (BESS) dispatch signals.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.