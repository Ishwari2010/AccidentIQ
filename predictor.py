# predictor.py
import pandas as pd
import numpy as np


def run_prediction(inputs: dict, model, scaler, label_encoders: dict, feature_order: list):
    """
    Encode inputs using saved label encoders, apply scaler if present,
    then return (predicted_class, probability_array).

    Parameters
    ----------
    inputs        : dict of raw user selections from the form
    model         : loaded sklearn/xgboost model
    scaler        : loaded StandardScaler (or None)
    label_encoders: dict  {column_name: fitted LabelEncoder}
    feature_order : list of column names in training order
    """
    # Build single-row DataFrame in training column order
    row = {}
    for feat in feature_order:
        row[feat] = [inputs.get(feat, "Unknown")]

    df = pd.DataFrame(row)

    # ── Encode categorical columns ────────────────────────────────────────────
    for col in df.columns:
        if col == "Number_of_vehicles_involved":
            df[col] = df[col].astype(int)
            continue

        le = label_encoders.get(col)
        if le is not None:
            val = df[col].iloc[0]
            # Handle unseen labels gracefully
            if val not in le.classes_:
                # Fall back to the first class (usually the most common)
                val = le.classes_[0]
            df[col] = le.transform([val])
        else:
            # No encoder available — try integer conversion, else 0
            try:
                df[col] = df[col].astype(int)
            except Exception:
                df[col] = 0

    # ── Scale if scaler is provided ───────────────────────────────────────────
    if scaler is not None:
        try:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=feature_order)
        except Exception:
            pass  # if scaler fails (column mismatch), proceed unscaled

    # ── Predict ───────────────────────────────────────────────────────────────
    pred_class = int(model.predict(df)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
    else:
        # For models without predict_proba (e.g. SVM without probability=True)
        proba = np.zeros(3)
        proba[pred_class] = 1.0

    # Ensure 3 classes always present
    if len(proba) < 3:
        full_proba = np.zeros(3)
        for i, p in enumerate(proba):
            full_proba[i] = p
        proba = full_proba

    return pred_class, proba
