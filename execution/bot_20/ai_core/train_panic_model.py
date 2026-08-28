import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

REGIME_CSV = "analysis/cycle_regimes.csv"
MODEL_DIR = "ai_core/models"
MODEL_PATH = os.path.join(MODEL_DIR, "panic_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "panic_model_meta.json")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Regime CSV not found: {path}")
    df = pd.read_csv(path)
    if "exit_reason" not in df.columns or "wallet_pnl" not in df.columns:
        raise ValueError("Expected columns 'exit_reason' and 'wallet_pnl' in regime CSV.")
    return df


def build_dataset(df: pd.DataFrame):
    # Focus ONLY on panic_bottom cycles
    df_pb = df[df["exit_reason"] == "panic_bottom"].copy()
    if df_pb.empty:
        raise ValueError("No panic_bottom cycles found. Cannot train panic model.")

    # Features we will use
    feature_cols = ["ema_gap_pct", "ema_slope", "volatility", "duration"]
    # 'duration' in our CSV is the column named 'duration'
    if "duration" not in df_pb.columns:
        # fallback: try 'duration_minutes'
        if "duration_minutes" in df_pb.columns:
            df_pb["duration"] = df_pb["duration_minutes"]
        else:
            df_pb["duration"] = np.nan

    # Drop rows with missing required features
    df_pb = df_pb.dropna(subset=feature_cols + ["wallet_pnl"])

    X = df_pb[feature_cols].values.astype(float)
    y = (df_pb["wallet_pnl"] > 0).astype(int).values  # 1 = profitable panic, 0 = bad panic

    return X, y, feature_cols, df_pb


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y)) > 1 else None,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }

    return clf, metrics


def main():
    print("=== Training Panic Model (AI Engine v2) ===")

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(REGIME_CSV)
    X, y, feature_cols, df_pb = build_dataset(df)

    print(f"[INFO] panic_bottom samples: {len(y)}")
    print(f"[INFO] positive panics (profit): {int(y.sum())}")
    print(f"[INFO] negative panics (loss): {int((y == 0).sum())}")

    clf, metrics = train_model(X, y)

    joblib.dump(clf, MODEL_PATH)
    print(f"[OK] Saved model → {MODEL_PATH}")

    meta = {
        "feature_cols": feature_cols,
        "metrics": metrics,
        "n_samples": int(len(y)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved meta → {META_PATH}")

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
