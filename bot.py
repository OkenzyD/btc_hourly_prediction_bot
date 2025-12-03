
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
XGB_MODEL_PATH      = f"{MODEL_DIR}/{BASE_NAME}_HYBRID_L10_xgb.json"

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
        title = entry.title
        published = pd.to_datetime(entry.published)
        score = analyzer.polarity_scores(title)["compound"]
        rows.append([published, score])

    if not rows:
        return pd.DataFrame(columns=["hour", "sentiment"])

    df = pd.DataFrame(rows, columns=["published", "sentiment"])
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
# 4. PREDICTION PIPELINE
# ============================================================
def run_prediction_pipeline():
    btc_df = fetch_btc_hourly()
    sent_df = fetch_google_news_sentiment()

    merged = merge_sentiment_with_decay(btc_df, sent_df)
    merged = enrich_live_dataset(merged)

    import json

    with open(f"{MODEL_DIR}/{BASE_NAME}_feature_list.json") as f:
        feature_cols = json.load(f)
        live_features = merged[feature_cols].iloc[-LOOKBACK:]
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    X_live_scaled = feature_scaler.transform(live_features)
    X_live_gru = X_live_scaled.reshape(1, LOOKBACK, len(feature_cols))

    gru_model = load_model(GRU_MODEL_PATH)
    gru_pred_scaled = gru_model.predict(X_live_gru)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    gru_pred = target_scaler.inverse_transform(gru_pred_scaled)[0][0]

    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )
    
    xgb_model.load_model(XGB_MODEL_PATH)
    hybrid_input = np.hstack([X_live_scaled[-1], [gru_pred]]).reshape(1, -1)
    hybrid_pred = xgb_model.predict(hybrid_input)[0]

    current_price = merged["close"].iloc[-1]
    move_pct = (hybrid_pred - current_price) / current_price * 100

    return current_price, gru_pred, hybrid_pred, move_pct


# ============================================================
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
        f"GRU:     ${gru_pred:,.2f}\n"
        f"Hybrid:  ${hybrid_pred:,.2f}\n"
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
