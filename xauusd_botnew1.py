"""
XAU/USD Professional Signal Bot
=================================
Sends fully-formatted Telegram signals matching the Valis-style format:

  🔴 XAU/USD — SHORT (87%)  |  01 Apr 2026
  ━━━━━━━━━━━━━━━━━━━━

  ENTRY: 2318.50
  SL:    2326.50  (8p)
  TP:    2306.50  (12p)
  R:R    1:1.5
  📊 Close 50% at TP, move SL to entry on rest
  🌊 SWING | ⏰ 5-10 hrs

  📊 Confidence: 87%
  💎 EV: +0.72R
  ⚠️ Entry: C (74/100)

  ⏰ TIMING
  ⚠️ OKAY (55%)

  💡 Factors:
  • 📉 Downtrend
  • ⚠️ Bullish CHoCH
  • 🔥 Strong ADX bearish (38)
  • 📌 Pin Bar Bear
  • 💪 USD stronger than XAU

Requirements:
    pip install requests pandas ta

Setup:
    1. Create a Telegram bot via @BotFather → copy the token.
    2. Message your bot once, then open:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id inside the JSON response.
    3. Paste TELEGRAM_TOKEN and CHAT_ID below OR export them as env vars.
    4. (Optional) Add a free Alpha Vantage key at https://alphavantage.co
       for dedicated XAUUSD data. Without it the bot uses Yahoo Finance GC=F.

Run:
    python xauusd_bot.py
"""

import os
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

try:
    import ta
except ImportError:
    raise SystemExit("Missing dependency — run:  pip install requests pandas ta")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit here or set environment variables
# ═══════════════════════════════════════════════════════════════════════════════

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "8631002566:AAEzNkCuoAO_i2h6GtrvWp4rSeaVZRr1J9s")
CHAT_ID          = os.getenv("CHAT_ID",          "5851314699")
TWELVE_DATA_KEY  = os.getenv("TWELVE_DATA_KEY",  "07d82c2484224923b517b991cf2b2442")

CHECK_INTERVAL = 60      # check every 60 seconds
MIN_SIGNAL_GAP = 5       # minimum minutes between signals

# Risk settings (USD price distance)
TP_PIPS = 12.0
SL_PIPS  = 8.0

# Indicator periods
EMA_FAST   = 9
EMA_SLOW   = 21
ADX_PERIOD = 14
RSI_PERIOD = 14
RSI_OB     = 65    # overbought — sell confirmation
RSI_OS     = 35    # oversold  — buy confirmation

# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("xauusd_bot")


# ─── Data fetching ─────────────────────────────────────────────────────────────

def fetch_ohlcv_twelvedata(n: int = 80) -> pd.DataFrame:
    """Twelve Data — real spot XAU/USD 5-minute candles."""
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval=5min&outputsize={n}&apikey={TWELVE_DATA_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    df = df.drop(columns=["datetime"]).astype(float)
    return df.sort_index().tail(n)


def fetch_ohlcv_yahoo(n: int = 80) -> pd.DataFrame:
    """Yahoo Finance GC=F fallback."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=1d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(result["timestamp"], unit="s"),
    )
    return df.dropna().tail(n)


def fetch_ohlcv() -> pd.DataFrame:
    try:
        log.info("Fetching via Twelve Data (XAU/USD spot) ...")
        return fetch_ohlcv_twelvedata()
    except Exception as e:
        log.warning("Twelve Data failed (%s) — falling back to Yahoo Finance ...", e)
        return fetch_ohlcv_yahoo()


# ─── Indicator helpers ─────────────────────────────────────────────────────────

def get_adx(df: pd.DataFrame) -> float:
    ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    return round(ind.adx().iloc[-1], 1)


def is_pin_bar(df: pd.DataFrame) -> bool:
    c    = df.iloc[-1]
    body = abs(c["close"] - c["open"])
    rng  = c["high"] - c["low"]
    if rng == 0:
        return False
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    return max(upper, lower) > 2 * body


def has_choch(df: pd.DataFrame) -> bool:
    """Detect a bullish CHoCH: recent price broke above a prior swing high."""
    closes = df["close"].tail(12)
    mid_high = closes.iloc[:6].max()
    recent   = closes.iloc[6:].max()
    return recent > mid_high


def trend_label(df: pd.DataFrame) -> str:
    fast = ta.trend.ema_indicator(df["close"], window=EMA_FAST).iloc[-1]
    slow = ta.trend.ema_indicator(df["close"], window=EMA_SLOW).iloc[-1]
    return "📉 Downtrend" if fast < slow else "📈 Uptrend"


# ─── Signal scoring ────────────────────────────────────────────────────────────

def score_signal(rsi: float, adx: float, pin_bar: bool, choch: bool, direction: str) -> dict:
    pts = 50

    # RSI confirmation
    if direction == "SHORT":
        if rsi > RSI_OB:       pts += 15
        elif rsi > 55:         pts += 7
    else:
        if rsi < RSI_OS:       pts += 15
        elif rsi < 45:         pts += 7

    # ADX trend strength
    if adx >= 35:              pts += 15
    elif adx >= 25:            pts += 8
    elif adx < 20:             pts -= 10

    # Pattern confirmation
    if pin_bar:                pts += 10

    # CHoCH warning on shorts
    if choch and direction == "SHORT":
        pts -= 8

    confidence = max(10, min(99, pts))

    # Expected value heuristic
    win_rate = confidence / 100
    rr_ratio  = TP_PIPS / SL_PIPS
    ev        = round(win_rate * rr_ratio - (1 - win_rate), 2)

    # Entry grade
    if confidence >= 80:   grade = f"A ({confidence}/100)"
    elif confidence >= 65: grade = f"B ({confidence}/100)"
    elif confidence >= 50: grade = f"C ({confidence}/100)"
    else:                  grade = f"D ({confidence}/100)"

    # Session timing
    hour = datetime.utcnow().hour
    if 7 <= hour <= 17:
        timing, t_icon = "GOOD (80%)",  "✅"
    elif 6 <= hour <= 20:
        timing, t_icon = "OKAY (50%)",  "⚠️"
    else:
        timing, t_icon = "POOR (30%)",  "🔴"

    return {
        "confidence": confidence,
        "ev":          ev,
        "grade":       grade,
        "timing":      timing,
        "timing_icon": t_icon,
    }


# ─── Core signal logic ─────────────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame) -> dict | None:
    df = df.copy()
    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["rsi"]      = ta.momentum.rsi(df["close"], window=RSI_PERIOD)

    prev, curr = df.iloc[-2], df.iloc[-1]
    price   = round(curr["close"], 2)
    rsi_val = round(curr["rsi"], 1)

    cross_up   = prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
    cross_down = prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]

    if not (cross_up or cross_down):
        return None

    direction = "LONG" if cross_up else "SHORT"

    # Require some RSI alignment
    if direction == "LONG"  and rsi_val > RSI_OS + 15: return None
    if direction == "SHORT" and rsi_val < RSI_OB - 15: return None

    adx      = get_adx(df)
    pin_bar  = is_pin_bar(df)
    choch    = has_choch(df)
    scores   = score_signal(rsi_val, adx, pin_bar, choch, direction)

    if direction == "SHORT":
        tp = round(price - TP_PIPS, 2)
        sl = round(price + SL_PIPS, 2)
    else:
        tp = round(price + TP_PIPS, 2)
        sl = round(price - SL_PIPS, 2)

    sl_dist = round(abs(price - sl), 2)
    tp_dist = round(abs(price - tp), 2)
    rr      = round(tp_dist / sl_dist, 1)

    factors = [trend_label(df)]
    if choch:
        factors.append("⚠️ Bullish CHoCH")
    if adx >= 25:
        side = "bearish" if direction == "SHORT" else "bullish"
        factors.append(f"🔥 Strong ADX {side} ({adx})")
    if pin_bar:
        factors.append("📌 Pin Bar " + ("Bear" if direction == "SHORT" else "Bull"))
    factors.append("💪 USD stronger than XAU" if direction == "SHORT" else "💪 XAU momentum vs USD")

    return {
        "direction": direction,
        "entry":     price,
        "sl":        sl,
        "tp":        tp,
        "sl_dist":   sl_dist,
        "tp_dist":   tp_dist,
        "rr":        rr,
        "rsi":       rsi_val,
        "adx":       adx,
        "factors":   factors,
        **scores,
    }


# ─── Telegram ──────────────────────────────────────────────────────────────────

def format_message(sig: dict) -> str:
    direction  = sig["direction"]
    confidence = sig["confidence"]
    dot        = "🔴" if direction == "SHORT" else "🟢"
    ev_str     = ("+" if sig["ev"] >= 0 else "") + str(sig["ev"]) + "R"
    date_str   = datetime.utcnow().strftime("%d %b %Y")
    factors_str = "\n".join(f"• {f}" for f in sig["factors"])

    return (
        f"{dot} *XAU/USD — {direction} ({confidence}%)*\n"
        f"_{date_str}_\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"*ENTRY:* `{sig['entry']}`\n"
        f"*SL:*    `{sig['sl']}` ({sig['sl_dist']}p)\n"
        f"*TP:*    `{sig['tp']}` ({sig['tp_dist']}p)\n"
        f"*R:R*    1:{sig['rr']}\n"
        f"📊 _Close 50% at TP, move SL to entry on rest_\n"
        f"🌊 *SWING* | ⏰ 5-10 hrs\n"
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Confidence: {confidence}%*\n"
        f"💎 EV: {ev_str}\n"
        f"⚠️ Entry: {sig['grade']}\n"
        f"\n"
        f"⏰ *TIMING*\n"
        f"{sig['timing_icon']} {sig['timing']}\n"
        f"\n"
        f"💡 *Factors:*\n"
        f"{factors_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
    r.raise_for_status()
    log.info("Telegram message sent.")


# ─── Entry point ───────────────────────────────────────────────────────────────

def validate_config() -> None:
    if "YOUR_BOT_TOKEN" in TELEGRAM_TOKEN:
        raise SystemExit("❌  Set TELEGRAM_TOKEN before running.")
    if "YOUR_CHAT_ID" in CHAT_ID:
        raise SystemExit("❌  Set CHAT_ID before running.")


def main() -> None:
    validate_config()
    log.info("XAU/USD bot started. Interval: %ds, cooldown: %dm", CHECK_INTERVAL, MIN_SIGNAL_GAP)
    send_telegram(
        "🤖 *XAU/USD Signal Bot online*\n"
        "_Scanning every 60 seconds — 5 min cooldown between signals…_"
    )

    last_signal_time = datetime.utcnow() - timedelta(minutes=MIN_SIGNAL_GAP + 1)

    while True:
        try:
            df     = fetch_ohlcv()
            signal = compute_signal(df)

            now    = datetime.utcnow()
            gap_ok = (now - last_signal_time).total_seconds() >= MIN_SIGNAL_GAP * 60

            if signal and gap_ok:
                log.info(
                    "SIGNAL %s  entry=%s  conf=%s%%",
                    signal["direction"], signal["entry"], signal["confidence"]
                )
                send_telegram(format_message(signal))
                last_signal_time = now
            elif signal and not gap_ok:
                secs_left = int(MIN_SIGNAL_GAP * 60 - (now - last_signal_time).total_seconds())
                log.info("Signal suppressed — cooldown %ds remaining", secs_left)
            else:
                log.info("No signal. Price: %s", round(df["close"].iloc[-1], 2))

        except Exception as exc:
            log.error("Error: %s", exc)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
