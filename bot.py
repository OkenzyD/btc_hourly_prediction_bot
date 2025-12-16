
# ===============================================================
#  BTC HOURLY AI PREDICTION BOT — Render Deployment Version
#  Fetch → Sentiment → Feature Engineering → GRU → XGBoost Hybrid → Bluesky
# ===============================================================

import os
import requests
import pandas as pd
import numpy as np
import joblib
import feedparser
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tensorflow.keras.models import load_model
import ta
from atproto import Client
import xgboost as xgb
import time
import json

# -----------------------------
# CONFIG
# -----------------------------
BASE_NAME = "BTC_Merged_Hourly"
LOOKBACK = 10

# Model directory (local repo folder)
MODEL_DIR = os.environ.get("MODEL_DIR", ".")

FEATURE_SCALER_PATH = f"{MODEL_DIR}/{BASE_NAME}_FeatureScaler.pkl"
TARGET_SCALER_PATH  = f"{MODEL_DIR}/{BASE_NAME}_TargetScaler.pkl"
GRU_MODEL_PATH      = f"{MODEL_DIR}/{BASE_NAME}_GRU_target_close_L10.keras"
XGB_MODEL_PATH = f"{MODEL_DIR}/{BASE_NAME}_PCT_ERROR_HYBRID_L10_xgb.json"

# Bluesky credentials stored as Render environment variables
BSKY_HANDLE       = os.environ.get("BSKY_HANDLE")
BSKY_APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD")


# ============================================================
# 1. FETCH BTC HOURLY FROM COINBASE
# ============================================================
def fetch_btc_hourly():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    data = requests.get(url).json()

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Coinbase API returned no data")

    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]



# ============================================================
# 2. SENTIMENT: GOOGLE NEWS RSS (VADER)
# ============================================================
def fetch_google_news_sentiment():
    url = "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)

    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for entry in feed.entries:
        title = getattr(entry, "title", None)
        published_raw = getattr(entry, "published", None)

        if not title:
            continue

        # Safe, guarded datetime parsing
        published = pd.to_datetime(published_raw, errors="coerce")

        if pd.isna(published):
            continue  # skip invalid timestamps

        score = analyzer.polarity_scores(title)["compound"]
        rows.append([published, score])

    # If nothing valid → fallback neutral sentiment
    if not rows:
        print("[WARN] No usable sentiment rows — applying neutral sentiment.")
        now_hour = pd.Timestamp.utcnow().floor("H")
        return pd.DataFrame({"hour": [now_hour], "sentiment": [0.0]})

    # Build DataFrame
    df = pd.DataFrame(rows, columns=["published", "sentiment"])

    # Force datetime conversion again (extra safety)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")

    # Drop any leftover NaT
    df = df.dropna(subset=["published"])

    # If everything got removed → fallback
    if df.empty:
        now_hour = pd.Timestamp.utcnow().floor("H")
        return pd.DataFrame({"hour": [now_hour], "sentiment": [0.0]})

    # Safe to use .dt now
    df["hour"] = df["published"].dt.floor("H")

    return df.groupby("hour")["sentiment"].mean().reset_index()

    df["hour"] = df["published"].dt.floor("H")

    return df.groupby("hour")["sentiment"].mean().reset_index()


def merge_sentiment_with_decay(btc_df, sent_df, decay=0.97):
    sent_df = sent_df.rename(columns={"sentiment": "sentiment_hourly"})
    merged = btc_df.merge(sent_df, left_on="timestamp", right_on="hour", how="left")
    merged = merged.drop(columns=["hour"])

    merged["sentiment_hourly"] = merged["sentiment_hourly"].ffill().fillna(0)
    sent_hours = set(sent_df["hour"])

    for i in range(1, len(merged)):
        ts = merged.loc[i, "timestamp"]
        if ts not in sent_hours:
            merged.loc[i, "sentiment_hourly"] = merged.loc[i - 1, "sentiment_hourly"] * decay

    return merged


    


# ============================================================
# 3. FEATURE ENGINEERING (MATCHES DISSERTATION)
# ============================================================
def enrich_live_dataset(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # ============================================================
    #  Ensure 'sentiment_hourly' ALWAYS exists BEFORE shifting
    # ============================================================
    if "sentiment_hourly" not in df.columns:
        print("[WARN] sentiment_hourly missing — injecting neutral 0")
        df["sentiment_hourly"] = 0.0

    df["sentiment_hourly"] = (
        df["sentiment_hourly"]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .astype(float)
    )

    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["ema_9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()

    df["sma_50"] = ta.trend.SMAIndicator(df["close"], 50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["close"], 200).sma_indicator()

    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_high"] - df["bb_low"]

    df["atr_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(df["high"], df["low"], df["close"], df["volume"], 14).volume_weighted_average_price()

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["rolling_volatility"] = df["close"].rolling(24, min_periods=5).std()
    df["rolling_volume"] = df["volume"].rolling(24, min_periods=5).mean()
    df["price_amplitude"] = (df["high"] - df["low"]) / df["close"]

    shift_features = [
        "rsi_14","macd","macd_signal","macd_diff","ema_9","ema_21",
        "sma_50","sma_200","bb_high","bb_low","bb_width","atr_14",
        "obv","vwap","rolling_volatility","rolling_volume",
        "price_amplitude","sentiment_hourly"
    ]
    df[shift_features] = df[shift_features].shift(1)

    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_close_{lag}"] = df["close"].shift(lag)
        df[f"lag_return_{lag}"] = df["close"].pct_change(lag)

    return df



# ============================================================
# 4. PREDICTION PIPELINE (DELTA HYBRID — FIXED & CLEAN)
# ============================================================

def run_prediction_pipeline():
    # -------------------------------------------------------
    # 1) Fetch data
    # -------------------------------------------------------
    btc_df = fetch_btc_hourly()
    sent_df = fetch_google_news_sentiment()

    merged = merge_sentiment_with_decay(btc_df, sent_df)
    merged = enrich_live_dataset(merged)

    # -------------------------------------------------------
    # 2) Load DELTA feature list
    # -------------------------------------------------------
    with open(f"{MODEL_DIR}/{BASE_NAME}_PCT_ERROR_XGB_feature_list.json") as f:
        feature_cols = json.load(f)

    # First 40 are the tabular features, last one is "gru_pred_raw"
    base_features = feature_cols[:-1]

    # -------------------------------------------------------
    # 3) Make sure sentiment_hourly exists & is numeric
    #    (extra safety – avoids KeyError when selecting base_features)
    # -------------------------------------------------------
    if "sentiment_hourly" not in merged.columns:
        print("[WARN] sentiment_hourly missing in merged — injecting neutral 0")
        merged["sentiment_hourly"] = 0.0

    merged["sentiment_hourly"] = (
        merged["sentiment_hourly"]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .astype(float)
    )

    # -------------------------------------------------------
    # 4) Select last LOOKBACK rows for GRU / scaler
    # -------------------------------------------------------
    live_features = merged[base_features].iloc[-LOOKBACK:].copy()
    live_features = live_features.fillna(0).replace([np.inf, -np.inf], 0)

    # -------------------------------------------------------
    # 5) Load scaler and enforce exact feature set / order
    # -------------------------------------------------------
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    required_cols = list(feature_scaler.feature_names_in_)

    # Ensure every feature the scaler expects exists
    for col in required_cols:
        if col not in live_features.columns:
            print(f"[WARN] Missing feature '{col}' → inserting default 0")
            live_features[col] = 0.0

    # Reorder columns to match scaler training order
    live_features = live_features[required_cols]

    # Scale
    X_live_scaled = feature_scaler.transform(live_features)

    # GRU expects shape: (1, LOOKBACK, n_features=40)
    X_live_gru = X_live_scaled.reshape(1, LOOKBACK, X_live_scaled.shape[1])

    # -------------------------------------------------------
    # 6) Run GRU model (scaled → unscaled price)
    # -------------------------------------------------------
    gru_model = load_model(GRU_MODEL_PATH)
    gru_pred_scaled = float(gru_model.predict(X_live_gru)[0][0])

    target_scaler = joblib.load(TARGET_SCALER_PATH)
    gru_pred_raw = float(target_scaler.inverse_transform([[gru_pred_scaled]])[0][0])
    
        # -------------------------------------------------------
    # TRUST GATE FOR GRU (REVISED: never freeze to current)
    current_price = float(merged["close"].iloc[-1])
    gru_deviation_pct = abs((gru_pred_raw - current_price) / current_price)

    if gru_deviation_pct > 0.04:  # 4%
    print(f"[WARN] GRU deviation too high ({gru_deviation_pct*100:.2f}%) — applying realism clamp instead of freezing")

    # build a small realistic move using volatility
    vol = float(merged["rolling_volatility"].iloc[-1])
    if np.isnan(vol) or vol <= 0:
        vol = current_price * 0.002  # 0.2% fallback

    raw_vol_pct = vol / current_price
    max_move_pct = float(np.clip(raw_vol_pct * 1.5, 0.0025, 0.02))  # 0.25%..2%

    direction = 1.0 if gru_pred_raw >= current_price else -1.0
    clamped_move_pct = direction * max_move_pct

    hybrid_pred = current_price * (1 + clamped_move_pct)
    move_pct = clamped_move_pct * 100

    return current_price, gru_pred_raw, hybrid_pred, move_pct

    # -------------------------------------------------------
    # 7) Load XGBoost delta-correction model
    # -------------------------------------------------------
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(XGB_MODEL_PATH)

    # -------------------------------------------------------
    # 8) Build XGB input exactly like in delta training
    #    - base features (scaled)
    #    - plus raw GRU prediction
    # -------------------------------------------------------
    # XGB delta expects EXACTLY 40 scaled features
    last_scaled = X_live_scaled[-1]
    live_tabular = pd.DataFrame([last_scaled], columns=required_cols)
    live_tabular["gru_pred_raw"] = gru_pred_raw

   
    
    print(f"[DEBUG] GRU raw prediction: {gru_pred_raw:.2f}")
    
    print("\n===== %-ERROR HYBRID INPUT (DEBUG) =====")
    print(live_tabular)
    print("=======================================\n")

    
  

     # -------------------------------------------------------
    # %-ERROR XGBOOST PREDICTION
    # -------------------------------------------------------
    pct_error_pred = float(xgb_model.predict(live_tabular)[0])
    
    # Safety clamp (percentage space, NOT price space)
    pct_error_pred = np.clip(pct_error_pred, -0.05, 0.05)  # ±5%
    
    # Final hybrid price (multiplicative correction)
    hybrid_pred = gru_pred_raw * (1 + pct_error_pred)


      # -------------------------------
    # REALISM GUARDRAIL 
    # -------------------------------
    current_price = float(merged["close"].iloc[-1])
    
    # Use your already-engineered volatility
    vol = float(merged["rolling_volatility"].iloc[-1])
    if np.isnan(vol) or vol <= 0:
        vol = current_price * 0.002  # fallback vol = 0.2%
    
    # Convert vol to a move band
    raw_vol_pct = vol / current_price            # e.g. 0.003 = 0.3%
    max_move_pct = raw_vol_pct * 1.5             # K=1.5 (tweakable)
    max_move_pct = float(np.clip(max_move_pct, 0.0025, 0.02))  # 0.25% .. 2.0%
    
    # Clamp your model output to realistic hourly move
    raw_move_pct = (hybrid_pred - current_price) / current_price
    clamped_move_pct = float(np.clip(raw_move_pct, -max_move_pct, max_move_pct))
    hybrid_pred = current_price * (1 + clamped_move_pct)
    
    # recompute move_pct after clamping
    move_pct = clamped_move_pct * 100

    # -------------------------------------------------------
    # 11) Safety / sanity check
    # -------------------------------------------------------




    
    print(
    f"[DEBUG] Current={current_price:.2f}, "
    f"GRU={gru_pred_raw:.2f}, "
    f"PctError={pct_error_pred:.4f}, "
    f"Hybrid={hybrid_pred:.2f}, "
    f"Move={move_pct:.2f}%"
    )
   # if hybrid_pred <= 0 or abs(move_pct) > 30:
    #    print("[WARN] Unstable hybrid result — fallback to GRU only.")
     #   hybrid_pred = gru_pred_raw
    #    move_pct = (hybrid_pred - current_price) / current_price * 100

    return current_price, gru_pred_raw, hybrid_pred, move_pct

# ===========================================================
# 5. POST TO BLUESKY
# ============================================================
def post_to_bluesky(current_price, gru_pred, hybrid_pred, move_pct):
    if not BSKY_HANDLE or not BSKY_APP_PASSWORD:
        print("Bluesky credentials not set — skipping post")
        return

    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)

    direction = "↑" if move_pct > 0 else "↓"

    msg = (
        f"BTC Hourly Forecast\n"
        f"Current: ${current_price:,.2f}\n"
        f"GRU_pred1:     ${gru_pred:,.2f}\n"
        f"Hybrid_pred2:  ${hybrid_pred:,.2f}\n"
        f"Move:    {move_pct:.2f}% {direction}\n"
        f"#Bitcoin #AI #Crypto"
    )

    client.send_post(text=msg)
    print(" Posted to Bluesky.")


# ============================================================
# 6. MAIN
# ============================================================


if __name__ == "__main__":
    print("===== Cron bot started =====")
    time.sleep(2)

    print("Running prediction pipeline...")
    current_price, gru_pred, hybrid_pred, move_pct = run_prediction_pipeline()

    print("Prediction OK, posting to Bluesky...")
    post_to_bluesky(current_price, gru_pred, hybrid_pred, move_pct)

    print("===== Cron bot finished =====")
