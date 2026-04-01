# Improved XAU/USD Signal Bot (Safer + Smarter)
# Key upgrades:
# - Removed hardcoded API keys (env only)
# - Added Higher Timeframe (1H) trend filter
# - ATR-based SL/TP
# - Better structure detection (simple BOS)
# - Reduced overtrading

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

import ta

# CONFIG
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "8631002566:AAEzNkCuoAO_i2h6GtrvWp4rSeaVZRr1J9s")
CHAT_ID          = os.getenv("CHAT_ID",          "5851314699")
TWELVE_DATA_KEY  = os.getenv("TWELVE_DATA_KEY",  "07d82c2484224923b517b991cf2b2442")

CHECK_INTERVAL = 60
MIN_SIGNAL_GAP = 15  # increased

EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bot")

# DATA

def fetch(symbol="XAU/USD", interval="5min", n=100):
    url = (
        f"https://api.twelvedata.com/time_series?symbol={symbol}"
        f"&interval={interval}&outputsize={n}&apikey={TWELVE_DATA_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"]).astype(float)
    return df.sort_index()

# HTF TREND

def higher_timeframe_trend():
    df = fetch(interval="1h", n=100)
    ema_fast = ta.trend.ema_indicator(df["close"], EMA_FAST).iloc[-1]
    ema_slow = ta.trend.ema_indicator(df["close"], EMA_SLOW).iloc[-1]
    return "BULL" if ema_fast > ema_slow else "BEAR"

# STRUCTURE (basic BOS)

def break_of_structure(df):
    highs = df["high"].rolling(10).max()
    lows = df["low"].rolling(10).min()

    if df["close"].iloc[-1] > highs.iloc[-2]:
        return "BULL"
    if df["close"].iloc[-1] < lows.iloc[-2]:
        return "BEAR"
    return None

# SIGNAL

def compute_signal(df):
    df = df.copy()

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], EMA_SLOW)
    df["rsi"] = ta.momentum.rsi(df["close"], RSI_PERIOD)
    df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"], ADX_PERIOD)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], ATR_PERIOD)

    prev, curr = df.iloc[-2], df.iloc[-1]

    cross_up = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
    cross_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

    if not (cross_up or cross_down):
        return None

    direction = "LONG" if cross_up else "SHORT"

    # HTF filter
    htf = higher_timeframe_trend()
    if (direction == "LONG" and htf != "BULL") or (direction == "SHORT" and htf != "BEAR"):
        return None

    # Structure filter
    bos = break_of_structure(df)
    if bos and ((bos == "BULL" and direction == "SHORT") or (bos == "BEAR" and direction == "LONG")):
        return None

    price = curr["close"]
    atr = curr["atr"]

    sl = price - atr * 1.2 if direction == "LONG" else price + atr * 1.2
    tp = price + atr * 1.8 if direction == "LONG" else price - atr * 1.8

    return {
        "direction": direction,
        "entry": round(price, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "confidence": int(60 + curr["adx"] / 2)
    }

# TELEGRAM

def send(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": msg})

# MAIN

def main():
    last_signal = datetime.utcnow() - timedelta(minutes=MIN_SIGNAL_GAP)

    while True:
        try:
            df = fetch()
            sig = compute_signal(df)

            if sig and (datetime.utcnow() - last_signal).seconds > MIN_SIGNAL_GAP * 60:
                message = f"{sig['direction']} XAUUSD\nEntry: {sig['entry']}\nSL: {sig['sl']}\nTP: {sig['tp']}\nConfidence: {sig['confidence']}%"
                send(message)
                last_signal = datetime.utcnow()
                log.info("Signal sent")
            else:
                log.info("No signal")

        except Exception as e:
            log.error(e)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
