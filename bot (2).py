


# ============================================================
# LIVE BTC HOURLY PREDICTION BOT (TFLITE VERSION)
# Fetch → Sentiment → Feature Eng → TFLite GRU → XGBoost → Bluesky
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
import joblib
import feedparser
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import ta
from atproto import Client
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

BASE_NAME = "BTC_Merged_Hourly"
LOOKBACK = 10

MODEL_DIR = os.environ.get("MODEL_DIR", ".")   # In Render/GitHub root

FEATURE_SCALER_PATH = os.path.join(MODEL_DIR, f"{BASE_NAME}_FeatureScaler.pkl")
TARGET_SCALER_PATH  = os.path.join(MODEL_DIR, f"{BASE_NAME}_TargetScaler.pkl")
TFLITE_MODEL_PATH   = os.path.join(MODEL_DIR, f"{BASE_NAME}_GRU_target_close_L10.tflite")
XGB_MODEL_PATH      = os.path.join(MODEL_DIR, f"{BASE_NAME}_HYBRID_L10_xgb.pkl")

# Bluesky credentials (must be set in Render Dashboard)
BSKY_HANDLE       = os.environ.get("BSKY_HANDLE")
BSKY_APP_PASSWORD = os.environ.get("BSKY_APP_PASSWORD")


# ============================================================
# FETCH BTC HOURLY — COINBASE
# ============================================================

def fetch_btc_hourly():
    """Fetch BTC-USD hourly OHLCV data from Coinbase."""
    print("Fetching BTC hourly from Coinbase...")

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
    data = requests.get(url).json()

    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Coinbase API returned invalid data")

    df = pd.DataFrame(
        data,
        columns=["time", "low", "high", "open", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print("✓ BTC hourly:", df.shape)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ============================================================
# GOOGLE NEWS SENTIMENT (VADER) + DECAY
# ============================================================

def fetch_google_news_sentiment():
    url = "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for entry in feed.entries:
        published = pd.to_datetime(entry.published)
        score = analyzer.polarity_scores(entry.title)["compound"]
        rows.append([published, score])

    if not rows:
        return pd.DataFrame(columns=["hour", "sentiment"])

    df = pd.DataFrame(rows, columns=["published", "sentiment"])
    df["hour"] = df["published"].dt.floor("H")

    grouped = df.groupby("hour")["sentiment"].mean().reset_index()
    print("✓ Sentiment rows:", grouped.shape)
    return grouped


def merge_sentiment_with_decay(btc_df, sent_df, decay=0.97):
    """Merge sentiment + apply exponential decay (matches your dissertation)."""

    if sent_df.empty:
        btc_df["sentiment_hourly"] = 0.0
        return btc_df

    sent_df = sent_df.rename(columns={"sentiment": "sentiment_hourly"})

    merged = btc_df.merge(
        sent_df[["hour", "sentiment_hourly"]],
        left_on="timestamp",
        right_on="hour",
        how="left"
    )
    merged.drop(columns=["hour"], inplace=True)

    merged["sentiment_hourly"] = merged["sentiment_hourly"].ffill().fillna(0.0)

    sent_hours = set(sent_df["hour"])

    for i in range(1, len(merged)):
        ts = merged.loc[i, "timestamp"]
        if ts not in sent_hours:
            merged.loc[i, "sentiment_hourly"] = merged.loc[i - 1, "sentiment_hourly"] * decay

    return merged


# ============================================================
# FEATURE ENGINEERING — EXACT REPLICA OF DISSERTATION
# ============================================================

def enrich_live_dataset(df):
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # Technical indicators
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

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], 14
    ).average_true_range()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], df["volume"]
    ).on_balance_volume()

    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"], 14
    ).volume_weighted_average_price()

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Volatility / liquidity
    df["rolling_volatility"] = df["close"].rolling(24, min_periods=5).std()
    df["rolling_volume"] = df["volume"].rolling(24, min_periods=5).mean()
    df["price_amplitude"] = (df["high"] - df["low"]) / df["close"]

    # Shift indicators (no lookahead)
    indicators = [
        "rsi_14", "macd", "macd_signal", "macd_diff",
        "ema_9", "ema_21", "sma_50", "sma_200",
        "bb_high", "bb_low", "bb_width",
        "atr_14", "obv", "vwap",
        "rolling_volatility", "rolling_volume",
        "price_amplitude",
        "sentiment_hourly"
    ]

    df[indicators] = df[indicators].shift(1)

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"lag_close_{lag}"] = df["close"].shift(lag)
        df[f"lag_return_{lag}"] = df["close"].pct_change(lag)

    # Targets (unused)
    df["target_close"] = df["close"].shift(-1)
    df["target_return"] = df["close"].pct_change().shift(-1)

    return df


# ============================================================
# TFLITE GRU INFERENCE
# ============================================================

def gru_predict_tflite(X_input):
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    X_input = X_input.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], X_input)
    interpreter.invoke()

    gru_scaled = interpreter.get_tensor(output_details[0]['index'])

    target_scaler = joblib.load(TARGET_SCALER_PATH)
    gru_unscaled = target_scaler.inverse_transform(gru_scaled)[0][0]

    return gru_unscaled


# ============================================================
# RUN FULL PREDICTION PIPELINE
# ============================================================

def run_prediction_pipeline():
    print("\n=== Running BTC Forecast Pipeline ===")

    btc_df = fetch_btc_hourly()
    sent_raw = fetch_google_news_sentiment()

    merged = merge_sentiment_with_decay(btc_df, sent_raw)
    merged = enrich_live_dataset(merged)

    train_df = pd.read_csv(os.path.join(MODEL_DIR, f"{BASE_NAME}_Train_Scaled.csv"))
    feature_cols = [
        c for c in train_df.columns
        if c not in ["timestamp", "target_close", "target_return"]
    ]

    live = merged[feature_cols].iloc[-LOOKBACK:]
    scaler = joblib.load(FEATURE_SCALER_PATH)

    X_scaled = scaler.transform(live)
    X_gru = X_scaled.reshape(1, LOOKBACK, len(feature_cols))

    # Predict using TFLite GRU
    gru_pred = gru_predict_tflite(X_gru)

    # Hybrid prediction
    xgb_model = joblib.load(XGB_MODEL_PATH)
    hybrid_input = np.hstack([X_scaled[-1], gru_pred]).reshape(1, -1)
    hybrid_pred = xgb_model.predict(hybrid_input)[0]

    current_price = merged["close"].iloc[-1]
    move_pct = (hybrid_pred - current_price) / current_price * 100

    print("\n=======================")
    print(" LIVE PREDICTION RESULT")
    print("=======================")
    print("Current close:", current_price)
    print("GRU forecast:", gru_pred)
    print("Hybrid forecast:", hybrid_pred)
    print("Expected move %:", move_pct)

    return current_price, gru_pred, hybrid_pred, move_pct


# ============================================================
# BLUESKY POSTING
# ============================================================

def post_to_bluesky(current, gru_pred, hybrid_pred, move_pct):
    if not BSKY_HANDLE or not BSKY_APP_PASSWORD:
        print("Bluesky env vars not set — skipping post.")
        return

    client = Client()
    client.login(BSKY_HANDLE, BSKY_APP_PASSWORD)

    direction = "↑" if move_pct > 0 else "↓"

    msg = (
        f"BTC Hourly Forecast\n\n"
        f"Current Price: ${current:,.2f}\n"
        f"GRU Forecast: ${gru_pred:,.2f}\n"
        f"Hybrid Forecast: ${hybrid_pred:,.2f}\n"
        f"Expected Move: {move_pct:.2f}% {direction}\n\n"
        f"#Bitcoin #Crypto #AI"
    )

    client.send_post(text=msg)
    print(" Posted to Bluesky")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================

def main():
    try:
        current, gru_p, hybrid_p, move = run_prediction_pipeline()
        post_to_bluesky(current, gru_p, hybrid_p, move)
    except Exception as e:
        print(" Pipeline error:", e)


if __name__ == "__main__":
    main()
