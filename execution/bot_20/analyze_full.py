#!/usr/bin/env python3
"""
analyze_full.py

Advanced analyser for the Uniswap V3 LP bot.
- Reads cycle_history.jsonl and last_cycle_data.json
- Builds multi-timeframe features
- Produces heuristic scores and (optionally) trains a small ML model to predict profitable choices
- Emits ai_tuning.json with recommended params:
    - min_width_ticks, max_width_ticks
    - normal_band_usd, panic_band_usd
    - panic cooldown, min_cycle_minutes, drift threshold
- Safe: works without sklearn/internet; ML is optional.
"""

import os
import json
import math
import time
from datetime import datetime, timedelta
from collections import deque, defaultdict

DATA_DIR = os.path.dirname(__file__) or "."
CYCLE_HISTORY = os.path.join(DATA_DIR, "cycle_history.jsonl")
LAST_STATE = os.path.join(DATA_DIR, "last_cycle_data.json")
AI_TUNING = os.path.join(DATA_DIR, "ai_tuning.json")

# ---------- ML imports (optional) ----------
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import numpy as np
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ---------- Utilities ----------
def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # ignore bad lines
                continue
    return out

def load_state():
    if not os.path.exists(LAST_STATE):
        return {}
    try:
        with open(LAST_STATE, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except Exception:
        return {}

def write_ai_tuning(d):
    with open(AI_TUNING, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print(f"✅ Wrote AI tuning to {AI_TUNING}")

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else 0.0

# ---------- Feature engineering ----------
def build_cycle_features(cycles):
    """
    Input: list of cycle dicts (from cycle_history.jsonl)
    Returns: list of feature dicts, aligned with cycles (same length)
    Feature examples:
      - duration_min
      - pnl_position_usd (pos only)
      - net_pnl_usd (after gas)
      - entry_price, exit_price, price_return_pct
      - width_ticks (if present)
      - fees_usd (computed)
      - recent_vols: realized vol over last 1h/3h/12h windows (via cycle returns)
      - win (net_pnl_usd > 0)
    """
    # convert cycles to enriched list
    out = []
    # we will treat each cycle as an observation at its withdrawal time
    # build time-indexed list
    timeline = []
    for c in cycles:
        ts = None
        for k in ("timestamp_withdraw", "timestamp_rebalance", "timestamp_create"):
            if k in c and c[k]:
                try:
                    ts = datetime.strptime(c[k], "%Y-%m-%d %H:%M:%S UTC")
                    break
                except Exception:
                    pass
        if ts is None:
            continue
        timeline.append((ts, c))
    timeline.sort(key=lambda x: x[0])

    # create simple price series from entry/exit prices in cycles
    price_points = []
    for ts,c in timeline:
        p_in = c.get("entry_eth_price")
        p_out = c.get("exit_eth_price")
        if p_in:
            try:
                price_points.append((ts - timedelta(milliseconds=1), float(p_in)))
            except: pass
        if p_out:
            try:
                price_points.append((ts, float(p_out)))
            except: pass
    price_points.sort()

    # helper to compute realized vol over a lookback window (seconds)
    def realized_vol(lookback_seconds, cur_ts):
        # compute absolute returns from price_points inside window
        start = cur_ts - timedelta(seconds=lookback_seconds)
        pts = [p for t,p in price_points if start <= t <= cur_ts]
        if len(pts) < 2:
            return 0.0
        # simple returns series
        rets = []
        for i in range(1, len(pts)):
            a = pts[i-1]
            b = pts[i]
            if a>0:
                rets.append(abs((b - a)/a))
        if not rets:
            return 0.0
        # annualize-ish: scale to per-3s tick is irrelevant, we return fraction
        return sum(rets)/len(rets)

    # walk timeline and produce features
    window_sizes = [60*10, 60*30, 60*60, 60*60*3]  # 10m,30m,1h,3h
    recent_returns = deque(maxlen=50)
    for ts,c in timeline:
        f = {}
        f["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        f["duration_min"] = float(c.get("duration_minutes") or 0.0)
        # net pnl (if present prefer net)
        net = c.get("net_pnl_usd")
        if net is None:
            net = c.get("net_usd") or c.get("pnl_position_usd") or c.get("pnl_usd") or 0.0
        f["net_pnl"] = float(net)
        f["pnl_position_usd"] = float(c.get("pnl_position_usd") or 0.0)
        f["entry_price"] = float(c.get("entry_eth_price") or 0.0)
        f["exit_price"] = float(c.get("exit_eth_price") or 0.0)
        # return over cycle (exit/entry -1)
        try:
            if f["entry_price"]>0:
                f["price_return_pct"] = (f["exit_price"] - f["entry_price"]) / f["entry_price"]
            else:
                f["price_return_pct"] = 0.0
        except:
            f["price_return_pct"] = 0.0

        f["width_ticks"] = int(c.get("range_width") or 0)
        f["entry_usdc"] = float(c.get("entry_usdc") or 0.0)
        f["entry_weth"] = float(c.get("entry_weth") or 0.0)
        f["fees_usd"] = float(c.get("fees_usd") or 0.0)

        # derived
        f["win"] = 1 if f["net_pnl"] > 0 else 0
        recent_returns.append(f["price_return_pct"])

        # multi-window vol features (from price_points)
        for w in window_sizes:
            f[f"rv_{w}s"] = realized_vol(w, ts)

        # simple momentum from recent_returns
        if len(recent_returns) >= 3:
            f["recent_mom"] = sum(list(recent_returns)[-3:])
        else:
            f["recent_mom"] = sum(recent_returns)

        out.append(f)

    return out

# ---------- Heuristic scoring ----------
def heuristic_score(curr_state, features_list):
    """
    Build an environment score and recommended config using deterministic heuristics.
    Returns dict with recommendations and a 'score' for confidence.
    """
    recent = features_list[-20:] if len(features_list) >= 1 else features_list
    if not recent:
        # default safe profile
        return {
            "score": 0.0,
            "recommend": {
                "range": {"min_width_ticks": 60, "max_width_ticks": 120},
                "usd_band": {"normal_band_usd": 40, "panic_band_usd": 70},
                "panic": {"panic_extra_usd": 10, "cooldown_minutes": 60},
                "rebalance": {"min_cycle_minutes": 60, "drift_threshold": 0.12}
            },
            "reason": "no-history-fallback"
        }

    # compute aggregated stats
    avg_net = safe_mean([f["net_pnl"] for f in recent])
    win_rate = safe_mean([f["win"] for f in recent])
    avg_dur = safe_mean([f["duration_min"] for f in recent])
    avg_fee_eff = safe_mean([(f["fees_usd"] / (abs(f["net_pnl"]) + 1e-9)) if f["net_pnl"]!=0 else 0.0 for f in recent])

    # vol estimates (use 1h and 3h realized volumes)
    rv_1h = safe_mean([f.get("rv_3600s",0.0) for f in recent])
    rv_3h = safe_mean([f.get("rv_10800s",0.0) for f in recent])
    vol = max(rv_1h, rv_3h)

    # heuristic rules to select widths:
    # - low vol & decent win rate -> tighten
    # - high vol -> widen
    # - if avg duration long and pnl positive -> increase min_cycle to reduce churn
    # map vol to width  (ticks)
    if vol < 0.0005:
        min_w, max_w = 40, 80
    elif vol < 0.002:
        min_w, max_w = 60, 100
    else:
        min_w, max_w = 80, 140

    # center bias: if recent_mom positive, bias range slightly to the upside (1-2%)
    mom = safe_mean([f.get("recent_mom",0.0) for f in recent])
    center_bias = 0.0
    if mom > 0.0:
        center_bias = 0.002  # +0.2% bias
    elif mom < 0.0:
        center_bias = -0.002

    # rebalance / cooldown tuning
    if avg_dur > 90:
        min_cycle = 90
    elif avg_dur > 45:
        min_cycle = 60
    else:
        min_cycle = 30

    panic_cooldown = max(20, int(min_cycle * 1.0))

    # drift threshold: wider bands allow larger drift
    drift = 0.10 if vol < 0.0008 else 0.14 if vol < 0.002 else 0.2

    # score confidence
    score = min(1.0, max(0.0, 0.5 + (win_rate-0.5) + (avg_net/50.0)))

    recommend = {
        "range": {"min_width_ticks": int(min_w), "max_width_ticks": int(max_w), "center_bias": center_bias},
        "usd_band": {"normal_band_usd": max(20, int(40 * (1 + vol*500))), "panic_band_usd": max(50, int(70 * (1 + vol*500)))},
        "panic": {"panic_extra_usd": 10, "panic_tick_buffer": 15, "cooldown_minutes": int(panic_cooldown)},
        "rebalance": {"drift_threshold": drift, "min_cycle_minutes": int(min_cycle), "max_rebalances_per_day": 6}
    }

    reason = (
        f"avg_net={avg_net:.2f}, win_rate={win_rate:.2f}, avg_dur={avg_dur:.1f}m, vol≈{vol:.5f}"
    )

    return {"score": score, "recommend": recommend, "reason": reason}

# ---------- Simple ML trainer (optional) ----------
def train_models(features):
    """
    Trains two small models if sklearn available:
      - classifier: predicts win (net_pnl>0)
      - regressor: predicts net_pnl
    Returns models or None.
    """
    if not SKLEARN_AVAILABLE:
        print("⚠️ sklearn not available — skipping ML training")
        return None, None

    # prepare dataset
    X = []
    y_clf = []
    y_reg = []
    for f in features:
        # simple features vector: duration, width_ticks, price_return_pct, rv_10m/30m/1h, recent_mom
        v = [
            f.get("duration_min",0.0),
            f.get("width_ticks",0),
            f.get("price_return_pct",0.0),
            f.get("rv_600s",0.0),
            f.get("rv_1800s",0.0),
            f.get("rv_3600s",0.0),
            f.get("recent_mom",0.0),
            f.get("fees_usd",0.0),
        ]
        X.append(v)
        y_clf.append(1 if f.get("net_pnl",0.0) > 0 else 0)
        y_reg.append(f.get("net_pnl",0.0))
    if len(X) < 8:
        print("⚠️ Not enough cycles for ML training (need >=8).")
        return None, None

    X = np.array(X)
    y_clf = np.array(y_clf)
    y_reg = np.array(y_reg)

    # random split
    Xtr, Xt, ytr, yt = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

    # classifier trained on win label
    clf.fit(X, y_clf)
    reg.fit(X, y_reg)

    print("✅ Trained ML models (clf/reg) on historical cycles.")
    return clf, reg

# ---------- Bandit-like chooser ----------
def choose_candidate_config(curr_state, heur, clf=None, reg=None):
    """
    Evaluate a set of candidate configs and score them using heuristics + optional ML predictions.
    Returns a final chosen config and ranked list.
    """
    # candidate widths to try (ticks)
    base_min = heur["recommend"]["range"]["min_width_ticks"]
    base_max = heur["recommend"]["range"]["max_width_ticks"]
    candidates = []
    # generate small set around base
    for w in [base_min-20, base_min, (base_min+base_max)//2, base_max, base_max+20]:
        w = max(20, int(w))
        candidates.append({"min_width_ticks": w, "max_width_ticks": max(w, w+40)})

    ranked = []
    # features for prediction: make a synthetic current-feature vector
    last = curr_state
    # build synthetic feature f
    fake_f = {
        "duration_min": last.get("last_duration_minutes") or 60.0,
        "width_ticks": last.get("range_width") or (base_min+base_max)//2,
        "price_return_pct": 0.0,
        "rv_600s": last.get("vol_ewma") or 0.0,
        "rv_1800s": last.get("vol_ewma") or 0.0,
        "rv_3600s": last.get("vol_ewma") or 0.0,
        "recent_mom": 0.0,
        "fees_usd": 0.0,
    }

    for c in candidates:
        # heuristic score base
        h_score = 1.0
        # penalise very tight for high vol
        vol = safe_mean([last.get("vol_ewma") or 0.0])
        if vol and c["min_width_ticks"] < 50 and vol > 0.001:
            h_score -= 0.35
        # small boost for middle-of-range widths
        if c["min_width_ticks"] > 60 and c["min_width_ticks"] < 110:
            h_score += 0.15

        ml_score = 0.0
        pred_pnl = 0.0
        if SKLEARN_AVAILABLE and clf is not None and reg is not None:
            # build X input
            X = [[
                fake_f["duration_min"],
                c["min_width_ticks"],
                fake_f["price_return_pct"],
                fake_f["rv_600s"],
                fake_f["rv_1800s"],
                fake_f["rv_3600s"],
                fake_f["recent_mom"],
                fake_f["fees_usd"],
            ]]
            try:
                win_prob = clf.predict_proba(X)[0][1] if hasattr(clf, "predict_proba") else clf.predict(X)[0]
                pred_pnl = float(reg.predict(X)[0])
                ml_score = 0.5*win_prob + 0.5* (1/(1+math.exp(-pred_pnl/10.0)))  # squash
            except Exception:
                ml_score = 0.0

        total = 0.6*h_score + 0.4*ml_score
        ranked.append({"candidate": c, "heur": h_score, "ml_score": ml_score, "pred_pnl": pred_pnl, "total": total})

    ranked.sort(key=lambda x: x["total"], reverse=True)
    chosen = ranked[0]
    return chosen, ranked

# ---------- Main entry ----------
def main():
    print("🔎 Loading cycle history...")
    cycles = load_jsonl(CYCLE_HISTORY)
    if not cycles:
        print("⚠️ No cycle history found. Please run some cycles first.")
    features = build_cycle_features(cycles)
    print(f"ℹ️ Loaded {len(features)} feature observations from history.")

    curr_state = load_state()
    # compute heuristics
    heur = heuristic_score(curr_state, features)
    print(f"ℹ️ Heuristic reason: {heur['reason']}; confidence {heur['score']:.2f}")

    # try ML training if available
    clf, reg = None, None
    if SKLEARN_AVAILABLE and len(features) >= 12:
        clf, reg = train_models(features)
    else:
        if not SKLEARN_AVAILABLE:
            print("ℹ️ sklearn not present — using heuristics only.")
        else:
            print("ℹ️ Not enough history for ML (need >=12 cycles).")

    chosen, ranked = choose_candidate_config(curr_state, heur, clf, reg)
    print("🔢 Top candidate config:", chosen)

    # Build final ai_tuning output with explanation
    ai_out = {
        "range": {
            "min_width_ticks": int(chosen["candidate"]["min_width_ticks"]),
            "max_width_ticks": int(chosen["candidate"]["max_width_ticks"]),
            "comment": "Chosen by hybrid analyser (heuristic + ML if available)."
        },
        "usd_band": heur["recommend"]["usd_band"],
        "panic": heur["recommend"]["panic"],
        "rebalance": heur["recommend"]["rebalance"],
        "meta": {
            "heur_confidence": heur["score"],
            "heur_reason": heur["reason"],
            "ml_used": bool(clf and reg),
            "ranked_candidates": ranked[:6],
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    }

    write_ai_tuning(ai_out)
    print("✅ Analysis complete — tune file written. Inspect ai_tuning.json for details.")

if __name__ == "__main__":
    main()
