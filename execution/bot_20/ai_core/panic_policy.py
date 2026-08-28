import os
import json
import joblib
import numpy as np

MODEL_DIR = "ai_core/models"
MODEL_PATH = os.path.join(MODEL_DIR, "panic_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "panic_model_meta.json")


class PanicDecisionModel:
    def __init__(self, model_path=MODEL_PATH, meta_path=META_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Panic model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Panic model meta not found: {meta_path}")

        self.model = joblib.load(model_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.metrics = meta.get("metrics", {})

    def predict_proba(self, ema_gap_pct, ema_slope, volatility, duration_minutes):
        """
        Returns probability that a panic exit would be PROFITABLE.
        """
        # Features order must match training
        feat_map = {
            "ema_gap_pct": float(ema_gap_pct),
            "ema_slope": float(ema_slope),
            "volatility": float(volatility),
            "duration": float(duration_minutes),
        }
        x = np.array([[feat_map[col] for col in self.feature_cols]], dtype=float)
        proba = self.model.predict_proba(x)[0, 1]  # prob of class 1 (profitable)
        return proba

    def should_panic(self, ema_gap_pct, ema_slope, volatility, duration_minutes, threshold=0.5):
        """
        Decide whether to trigger panic, given current features.
        Returns (bool, prob_win).
        threshold = minimum probability that panic is profitable.
        """
        p_win = self.predict_proba(ema_gap_pct, ema_slope, volatility, duration_minutes)
        return (p_win >= threshold), p_win


# Convenience function for use in your bot
_global_model = None

def should_panic_exit(ema_gap_pct, ema_slope, volatility, duration_minutes, threshold=0.5):
    global _global_model
    if _global_model is None:
        _global_model = PanicDecisionModel()
    return _global_model.should_panic(ema_gap_pct, ema_slope, volatility, duration_minutes, threshold)
