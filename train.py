import os
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from features import FEATURE_COLUMNS

DATA_PATH = "data/historical_matches.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "catboost_model.cbm")

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No training data at {DATA_PATH}. Run build_training_data.py first.")
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {len(df)} matches")
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["result"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training on {len(X_train)}, validating on {len(X_val)}...")
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        loss_function="MultiClass", eval_metric="Accuracy",
        random_seed=42, verbose=100, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\nValidation Accuracy: {round(acc * 100, 1)}%")
    print(classification_report(y_val, y_pred, target_names=["Home Win", "Draw", "Away Win"]))
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print("Commit model/catboost_model.cbm to GitHub.")

if __name__ == "__main__":
    train()
