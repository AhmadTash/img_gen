import joblib
import os
import pandas as pd
from typing import Optional
try:
    from .bounds import clamp_params
    from .train import MODEL_FILE, TARGET_COLUMNS, FEATURE_COLUMNS
except ImportError:
    from bounds import clamp_params
    from train import MODEL_FILE, TARGET_COLUMNS, FEATURE_COLUMNS

# Global cache for the loaded model
_model = None

def load_model():
    global _model
    if _model is not None:
        return _model
    
    if not os.path.exists(MODEL_FILE):
        return None
        
    try:
        _model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
        
    return _model

def predict_params(features: dict) -> Optional[dict]:
    """
    Predicts aesthetic parameters from visual features.
    Returns None if model is not loaded/trained.
    """
    model = load_model()
    if model is None:
        return None
        
    # Prepare input dataframe
    # Ensure correct order and presence of features
    try:
        # Create a single-row DataFrame
        df = pd.DataFrame([features])
        
        # Select only expected features, fill missing with 0 (shouldn't happen if properly extracted)
        X = df[FEATURE_COLUMNS] if set(FEATURE_COLUMNS).issubset(df.columns) else None
        
        if X is None:
            # Fallback: try to construct if keys are missing? 
            # Ideally extract_features guarantees keys.
            return None

        # Predict
        pred_array = model.predict(X) 
        # pred_array shape is (1, n_targets)
        
        # Map back to dict
        raw_pred = {}
        for idx, col in enumerate(TARGET_COLUMNS):
            raw_pred[col] = float(pred_array[0, idx])
            
        # Clamp
        final_params = clamp_params(raw_pred)
        return final_params

    except Exception as e:
        print(f"Prediction error: {e}")
        return None
