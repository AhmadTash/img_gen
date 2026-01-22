import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
try:
    from .dataset import load_dataset
except ImportError:
    from dataset import load_dataset

MODEL_FILE = os.path.join(os.path.dirname(__file__), "param_model.pkl")

# Targets we want to predict
# logic: mapped from constraints
# paint_thickness, messiness, text_wobble, shadow_opacity (as paint_opacity), blur_mix (as grain)
TARGET_COLUMNS = [
    "paint_thickness",
    "messiness",
    "text_wobble",
    "shadow_opacity",
    "blur_mix"
]

# Input features must match ml/features.py
FEATURE_COLUMNS = [
    "mean_brightness",
    "contrast",
    "mean_saturation",
    "warmth",
    "sharpness"
]

def train_model(dataset_path: str = None, model_path: str = MODEL_FILE) -> None:
    if dataset_path:
        df = load_dataset(dataset_path)
    else:
        df = load_dataset()

    if df is None or len(df) < 5:
        print("Not enough data to train (need at least 5 samples).")
        return

    # Filter columns that actually exist in the CSV
    # (Handling case where CSV might miss some columns if params changed)
    valid_targets = [t for t in TARGET_COLUMNS if t in df.columns]
    valid_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    if not valid_targets or not valid_features:
        print("Missing targets or features in dataset.")
        return

    X = df[valid_features]
    y = df[valid_targets]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)

    # Evaluate
    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. MAE on test set: {mae:.4f}")

    # Save
    joblib.dump(regr, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
