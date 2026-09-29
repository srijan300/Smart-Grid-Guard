import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(
    df, y_true, y_pred,
    contamination=None,        # if None, use score threshold instead
    score_percentile=98.5,     # top X% by anomaly score
    rolling_window=24          # for residual baseline (adjust to your data freq)
):
    """
    Contextual anomaly detection using Isolation Forest with error + calendar features.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing calendar features (dayofweek, weekend, holiday).
    y_true : array-like
        Actual demand values
    y_pred : array-like
        LSTM forecast values
    contamination : float or None
        Expected anomaly proportion (e.g., 0.02 = 2%). If None, use score_percentile.
    score_percentile : float
        Percentile threshold for anomaly scores (only used if contamination=None).
    rolling_window : int
        Window size for residual smoothing (captures seasonality).

    Returns
    -------
    anomalies : pd.DataFrame
        Rows flagged as anomalies
    iso : IsolationForest
        Trained model
    """

    # Ensure series
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    n_samples = len(y_pred)
    # Align y_true if full series was passed instead of pre-aligned series
    if len(y_true) > n_samples:
        offset = len(y_true) - n_samples
        y_true = y_true.iloc[offset:].reset_index(drop=True)
    else:
        offset = len(df) - n_samples if len(df) >= n_samples else 0

    errors = (y_true - y_pred).values.reshape(-1, 1)

    # ---- Rolling residuals ----
    rolling_mean = y_true.rolling(rolling_window, min_periods=1).mean()
    rolling_mean_pred = y_pred.rolling(rolling_window, min_periods=1).mean()
    residuals = (y_true - rolling_mean) - (y_pred - rolling_mean_pred)

    # ---- Build feature matrix ----
    feature_df = pd.DataFrame({
        "abs_error": np.abs(errors).ravel(),
        "signed_error": errors.ravel(),
        "residual": residuals.fillna(0).values
    })

    # ---- Calendar features (case-insensitive column search) ----
    col_map = {c.lower(): c for c in df.columns}
    if "dayofweek" in col_map:
        dow = df[col_map["dayofweek"]].iloc[-n_samples:].reset_index(drop=True).astype(int)
        feature_df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        feature_df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    
    for c_name in ["weekend", "holiday"]:
        if c_name in col_map:
            feature_df[c_name] = df[col_map[c_name]].iloc[-n_samples:].reset_index(drop=True).astype(int)

    # ---- Scale features ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    # ---- Train Isolation Forest ----
    iso = IsolationForest(
        contamination=contamination if contamination is not None else 'auto',
        n_estimators=300,
        random_state=42
    )
    iso.fit(X_scaled)

    # anomaly scores (higher = more anomalous)
    scores = -iso.decision_function(X_scaled)

    if contamination is not None:
        labels = iso.predict(X_scaled)
        is_anomaly = labels == -1
    else:
        threshold = np.percentile(scores, score_percentile)
        is_anomaly = scores >= threshold

    # ---- Build output ----
    feature_df["actual"] = y_true.values
    feature_df["predicted"] = y_pred.values
    feature_df["score"] = scores
    feature_df["is_anomaly"] = is_anomaly
    feature_df["index"] = np.arange(offset, offset + n_samples)

    anomalies = feature_df[feature_df["is_anomaly"]]

    return anomalies, iso
