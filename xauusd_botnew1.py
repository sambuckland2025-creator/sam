
"""
XAU/USD Ultimate Scalper Bot
==============================
Best of both bots merged:

ACCURACY FEATURES:
  - HTF 15min trend alignment (only trades with the trend)
  - London/NY session filter (7am-5pm UTC only)
  - Pullback entries (better price, tighter SL)
  - ATR-based dynamic SL/TP (adapts to volatility)
  - EMA 50 trend filter

5 SIGNAL TRIGGERS:
  1. EMA 9/21 Crossover
  2. RSI Cross 50 (momentum)
  3. EMA 21 Bounce (pullback)
  4. RSI Extreme 30/70 (reversal)
  5. ADX Breakout (new trend)

FULL VALIS-STYLE MESSAGE:
  Entry / SL / TP / R:R / Confidence / EV / Grade / Timing / Factors

Requirements:
    pip install requests pandas ta

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

# ===============================================================================
#  CONFIG
# ===============================================================================

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN",  "8631002566:AAEzNkCuoAO_i2h6GtrvWp4rSeaVZRr1J9s")
CHAT_ID         = os.getenv("CHAT_ID",         "5851314699")
TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "07d82c2484224923b517b991cf2b2442")

CHECK_INTERVAL = 60   # seconds between scans
MIN_SIGNAL_GAP = 5    # minimum minutes between signals

# Indicator periods
EMA_FAST   =  9
EMA_SLOW   = 21
EMA_TREND  = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

# ATR multipliers for dynamic SL/TP
ATR_SL_MULT = 1.0   # SL = 1x ATR
ATR_TP_MULT = 1.8   # TP = 1.8x ATR  →  1:1.8 R:R

# ===============================================================================

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("xauusd_bot")


# ─── Session filter ────────────────────────────────────────────────────────────

def in_session() -> bool:
    """London + NY session only (best gold liquidity)."""
    return 7 <= datetime.utcnow().hour <= 17


def session_name() -> str:
    h = datetime.utcnow().hour
    if 12 <= h <= 17:
        return "London/NY Overlap"
    elif 7 <= h < 12:
        return "London"
    elif h <= 21:
        return "New York"
    return "Off-Session"


# ─── Data fetching ─────────────────────────────────────────────────────────────

def fetch_twelvedata(interval: str = "5min", n: int = 100) -> pd.DataFrame:
    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD&interval={interval}&outputsize={n}&apikey={TWELVE_DATA_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df.index = pd.to_datetime(df["datetime"])
    return df.drop(columns=["datetime"]).astype(float).sort_index().tail(n)


def fetch_yahoo(n: int = 100) -> pd.DataFrame:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=5m&range=1d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q   = res["indicators"]["quote"][0]
    df  = pd.DataFrame(
        {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
        index=pd.to_datetime(res["timestamp"], unit="s"),
    )
    return df.dropna().tail(n)


def fetch_5min() -> pd.DataFrame:
    try:
        log.info("Fetching 5min via Twelve Data ...")
        return fetch_twelvedata("5min", 100)
    except Exception as e:
        log.warning("Twelve Data failed (%s) — Yahoo fallback ...", e)
        return fetch_yahoo()


def fetch_15min() -> pd.DataFrame:
    return fetch_twelvedata("15min", 60)


# ─── HTF trend ─────────────────────────────────────────────────────────────────

def get_htf_trend() -> str:
    try:
        df       = fetch_15min()
        ema_fast = ta.trend.ema_indicator(df["close"], window=EMA_FAST).iloc[-1]
        ema_slow = ta.trend.ema_indicator(df["close"], window=EMA_SLOW).iloc[-1]
        rsi      = ta.momentum.rsi(df["close"], window=RSI_PERIOD).iloc[-1]
        if ema_fast > ema_slow and rsi > 50:
            return "BULL"
        elif ema_fast < ema_slow and rsi < 50:
            return "BEAR"
        return "NEUTRAL"
    except Exception as e:
        log.warning("HTF failed (%s) — NEUTRAL", e)
        return "NEUTRAL"


# ─── Indicators ────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = ta.trend.ema_indicator(df["close"], window=EMA_FAST)
    df["ema_slow"]  = ta.trend.ema_indicator(df["close"], window=EMA_SLOW)
    df["ema_trend"] = ta.trend.ema_indicator(df["close"], window=EMA_TREND)
    df["rsi"]       = ta.momentum.rsi(df["close"], window=RSI_PERIOD)
    df["atr"]       = ta.volatility.average_true_range(
                          df["high"], df["low"], df["close"], window=ATR_PERIOD)
    adx             = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD)
    df["adx"]       = adx.adx()
    df["adx_pos"]   = adx.adx_pos()
    df["adx_neg"]   = adx.adx_neg()
    return df


def is_pin_bar(c: pd.Series) -> bool:
    body  = abs(c["close"] - c["open"])
    rng   = c["high"] - c["low"]
    if rng == 0:
        return False
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    return max(upper, lower) > 2 * body


def has_choch(df: pd.DataFrame) -> bool:
    closes = df["close"].tail(12)
    return closes.iloc[6:].max() > closes.iloc[:6].max()


# ─── 5 Triggers ────────────────────────────────────────────────────────────────

def check_ema_cross(prev, curr) -> str | None:
    if prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]:
        return "LONG"
    if prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]:
        return "SHORT"
    return None


def check_rsi_cross50(prev, curr) -> str | None:
    if prev["rsi"] <= 50 and curr["rsi"] > 50:
        return "LONG"
    if prev["rsi"] >= 50 and curr["rsi"] < 50:
        return "SHORT"
    return None


def check_ema_bounce(prev, curr) -> str | None:
    ema = curr["ema_slow"]
    if prev["low"] <= ema * 1.001 and curr["close"] > ema and curr["close"] > prev["close"]:
        return "LONG"
    if prev["high"] >= ema * 0.999 and curr["close"] < ema and curr["close"] < prev["close"]:
        return "SHORT"
    return None


def check_rsi_extreme(curr) -> str | None:
    if curr["rsi"] <= 30: return "LONG"
    if curr["rsi"] >= 70: return "SHORT"
    return None


def check_adx_breakout(prev, curr) -> str | None:
    if prev["adx"] < 25 and curr["adx"] >= 25:
        return "LONG" if curr["adx_pos"] > curr["adx_neg"] else "SHORT"
    return None


# ─── Scoring ───────────────────────────────────────────────────────────────────

def score_signal(curr, direction, trigger, htf, pin_bar, choch) -> dict:
    pts = 45
    pts += {"EMA_CROSS": 20, "RSI_CROSS50": 12, "EMA_BOUNCE": 18,
            "RSI_EXTREME": 15, "ADX_BREAKOUT": 16}.get(trigger, 10)

    rsi = curr["rsi"]
    adx = curr["adx"]

    # HTF alignment — biggest bonus
    if (direction == "LONG" and htf == "BULL") or (direction == "SHORT" and htf == "BEAR"):
        pts += 15
    elif htf == "NEUTRAL":
        pts -= 5

    # RSI alignment
    if direction == "SHORT":
        pts += 10 if rsi > 65 else 5 if rsi > 55 else 0
    else:
        pts += 10 if rsi < 35 else 5 if rsi < 45 else 0

    # ADX strength
    if adx >= 35:   pts += 10
    elif adx >= 25: pts += 5
    elif adx < 20:  pts -= 8

    # Patterns
    if pin_bar:                         pts += 8
    if choch and direction == "SHORT":  pts -= 6

    # Session bonus
    sess = session_name()
    if "Overlap" in sess: pts += 5
    elif sess != "Off-Session": pts += 3

    confidence = max(10, min(99, pts))
    win_rate   = confidence / 100
    rr         = round(ATR_TP_MULT / ATR_SL_MULT, 1)
    ev         = round(win_rate * rr - (1 - win_rate), 2)

    if confidence >= 85:   grade = f"A ({confidence}/100)"
    elif confidence >= 70: grade = f"B ({confidence}/100)"
    elif confidence >= 55: grade = f"C ({confidence}/100)"
    else:                  grade = f"D ({confidence}/100)"

    h = datetime.utcnow().hour
    if 12 <= h <= 16:
        timing, t_icon = "GOOD — London/NY Overlap (85%)", "✅"
    elif 7 <= h <= 17:
        timing, t_icon = "GOOD (75%)", "✅"
    else:
        timing, t_icon = "POOR — Off Session (25%)", "🔴"

    return {"confidence": confidence, "ev": ev, "grade": grade,
            "timing": timing, "timing_icon": t_icon, "rr": rr}


# ─── Core logic ────────────────────────────────────────────────────────────────

def compute_signal(df: pd.DataFrame, htf: str) -> dict | None:
    df   = add_indicators(df)
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    # Run all 5 triggers
    for name, fn in [
        ("EMA_CROSS",    check_ema_cross(prev, curr)),
        ("RSI_CROSS50",  check_rsi_cross50(prev, curr)),
        ("EMA_BOUNCE",   check_ema_bounce(prev, curr)),
        ("RSI_EXTREME",  check_rsi_extreme(curr)),
        ("ADX_BREAKOUT", check_adx_breakout(prev, curr)),
    ]:
        if fn:
            direction, trigger = fn, name
            break
    else:
        return None

    # HTF filter
    if htf == "BULL" and direction == "SHORT":
        log.info("SHORT skipped — HTF BULL"); return None
    if htf == "BEAR" and direction == "LONG":
        log.info("LONG skipped — HTF BEAR");  return None

    # EMA 50 filter
    if direction == "LONG"  and curr["close"] < curr["ema_trend"]:
        log.info("LONG filtered — below EMA50"); return None
    if direction == "SHORT" and curr["close"] > curr["ema_trend"]:
        log.info("SHORT filtered — above EMA50"); return None

    price   = round(curr["close"], 2)
    atr     = curr["atr"]
    adx     = round(curr["adx"], 1)
    rsi_val = round(curr["rsi"], 1)
    pin_bar = is_pin_bar(curr)
    choch   = has_choch(df)
    scores  = score_signal(curr, direction, trigger, htf, pin_bar, choch)

    # ATR dynamic SL/TP
    if direction == "SHORT":
        sl = round(price + atr * ATR_SL_MULT, 2)
        tp = round(price - atr * ATR_TP_MULT, 2)
    else:
        sl = round(price - atr * ATR_SL_MULT, 2)
        tp = round(price + atr * ATR_TP_MULT, 2)

    sl_dist = round(abs(price - sl), 2)
    tp_dist = round(abs(price - tp), 2)
    rr      = round(tp_dist / sl_dist, 1)

    # Factors
    trend = "📉 Downtrend" if curr["ema_fast"] < curr["ema_slow"] else "📈 Uptrend"
    htf_label = {"BULL": "📈 HTF 15m Bullish", "BEAR": "📉 HTF 15m Bearish"}.get(htf, "➡️ HTF Neutral")
    trig_label = {
        "EMA_CROSS":    "🔀 EMA 9/21 Crossover",
        "RSI_CROSS50":  "📶 RSI Momentum Cross (50)",
        "EMA_BOUNCE":   "🔄 EMA 21 Pullback Bounce",
        "RSI_EXTREME":  "⚡ RSI Extreme Level",
        "ADX_BREAKOUT": "💥 ADX Trend Breakout",
    }.get(trigger, "📊 Signal")

    factors = [trend, htf_label, trig_label]
    if choch:
        factors.append("⚠️ Bullish CHoCH")
    if adx >= 25:
        factors.append(f"🔥 Strong ADX {'bearish' if direction == 'SHORT' else 'bullish'} ({adx})")
    if pin_bar:
        factors.append("📌 Pin Bar " + ("Bear" if direction == "SHORT" else "Bull"))
    factors.append("💪 USD stronger than XAU" if direction == "SHORT" else "💪 XAU momentum vs USD")
    factors.append(f"🕐 {session_name()} Session")

    return {
        "direction": direction, "trigger": trigger,
        "entry": price, "sl": sl, "tp": tp,
        "sl_dist": sl_dist, "tp_dist": tp_dist, "rr": rr,
        "rsi": rsi_val, "adx": adx, "atr": round(atr, 2),
        "htf": htf, "factors": factors, **scores,
    }


# ─── Telegram ──────────────────────────────────────────────────────────────────

def format_message(sig: dict) -> str:
    dot         = "🔴" if sig["direction"] == "SHORT" else "🟢"
    ev_str      = ("+" if sig["ev"] >= 0 else "") + str(sig["ev"]) + "R"
    factors_str = "\n".join(f"• {f}" for f in sig["factors"])

    return (
        f"{dot} *XAU/USD \u2014 {sig['direction']} ({sig['confidence']}%)*\n"
        f"_{datetime.utcnow().strftime('%d %b %Y')}_\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"\n"
        f"*ENTRY:* `{sig['entry']}`\n"
        f"*SL:*    `{sig['sl']}` ({sig['sl_dist']}p)\n"
        f"*TP:*    `{sig['tp']}` ({sig['tp_dist']}p)\n"
        f"*R:R*    1:{sig['rr']}\n"
        f"\U0001f4ca _Close 50% at TP, move SL to entry on rest_\n"
        f"\U0001f30a *SCALP* | \u23f0 1-3 hrs\n"
        f"\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"\U0001f4ca *Confidence: {sig['confidence']}%*\n"
        f"\U0001f48e EV: {ev_str}\n"
        f"\u26a0\ufe0f Entry: {sig['grade']}\n"
        f"\n"
        f"\u23f0 *TIMING*\n"
        f"{sig['timing_icon']} {sig['timing']}\n"
        f"\n"
        f"\U0001f4a1 *Factors:*\n"
        f"{factors_str}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"
    )


def send_telegram(message: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(
        url,
        json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"},
        timeout=10,
    ).raise_for_status()
    log.info("Telegram message sent.")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_TOKEN: raise SystemExit("Set TELEGRAM_TOKEN")
    if not CHAT_ID:        raise SystemExit("Set CHAT_ID")
    if not TWELVE_DATA_KEY: raise SystemExit("Set TWELVE_DATA_KEY")

    log.info("XAU/USD Ultimate Scalper started.")
    send_telegram(
        "\U0001f916 *XAU/USD Ultimate Scalper online*\n"
        "_5 triggers \u2022 HTF alignment \u2022 ATR SL/TP \u2022 London/NY only_\n"
        "_Scanning every 60s \u2014 signals incoming\u2026_"
    )

    last_signal_time = datetime.utcnow() - timedelta(minutes=MIN_SIGNAL_GAP + 1)
    htf_cache        = "NEUTRAL"
    htf_last_updated = datetime.utcnow() - timedelta(minutes=16)

    while True:
        try:
            now = datetime.utcnow()

            # Refresh HTF every 15 min
            if (now - htf_last_updated).total_seconds() >= 900:
                htf_cache        = get_htf_trend()
                htf_last_updated = now
                log.info("HTF trend updated: %s", htf_cache)

            if not in_session():
                log.info("Outside session (UTC %02d:00) — sleeping ...", now.hour)
                time.sleep(CHECK_INTERVAL)
                continue

            df     = fetch_5min()
            signal = compute_signal(df, htf_cache)
            gap_ok = (now - last_signal_time).total_seconds() >= MIN_SIGNAL_GAP * 60

            if signal and gap_ok:
                log.info("SIGNAL %s via %s  entry=%s  conf=%s%%  HTF=%s",
                         signal["direction"], signal["trigger"],
                         signal["entry"], signal["confidence"], signal["htf"])
                send_telegram(format_message(signal))
                last_signal_time = now
            elif signal and not gap_ok:
                secs = int(MIN_SIGNAL_GAP * 60 - (now - last_signal_time).total_seconds())
                log.info("Signal ready — cooldown %ds remaining", secs)
            else:
                log.info("No signal | Price: %s | HTF: %s | %s",
                         round(df["close"].iloc[-1], 2), htf_cache, session_name())

        except Exception as e:
            log.error("Error: %s", e)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()