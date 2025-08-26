# app.py
# Streamlit Trading-Analyse – Single-File-App
# Funktionen: Kursdaten via yfinance, TA (SMA/EMA/RSI/MACD/Stoch/ATR),
# Unterstützungen/Widerstände, einfache News-/Sentiment-Auswertung via RSS (feedparser),
# strukturierte Ausgabe inkl. Empfehlung, SL/TP-Vorschlag und interaktiver Chart.

import math
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import re

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- UI/Seiten-Setup ----------
st.set_page_config(
    page_title="Trading-Analyse (Single-File)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MOBILE_HINT = """
**Tipp (Mobile):** Nutze das ⋮-Menü oben rechts → *View fullscreen*.
"""

st.markdown(
    """
    <style>
    /* Mobile: Buttons/Eingaben großflächig */
    .stButton>button {width:100%; padding:0.9rem 1rem; border-radius:14px; font-weight:600}
    .stTextInput>div>div>input, .stSelectbox>div>div>div {padding:0.8rem 0.8rem; border-radius:12px}
    .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1100px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Trading-Analyse (Aktien • Krypto • FX • Indizes)")
st.caption("Ein-Datei-App für Streamlit Cloud | Daten: yfinance | News: RSS/Feedparser")
st.info(MOBILE_HINT)

# ---------- Eingaben ----------
colA, colB = st.columns([1.2, 1])
with colA:
    ticker_input = st.text_input(
        "Asset/Symbol (z. B. AAPL, TSLA, BTC-USD, ETH-USD, ^NDX, EURUSD=X)",
        value="BTC-USD",
        help="Verwende Yahoo Finance Symbole.",
    )

with colB:
    timeframe = st.selectbox(
        "Zeitrahmen",
        options=["1h", "4h", "1d", "1W", "1M"],
        index=2,
        help="Wird auf period/interval von yfinance gemappt.",
    )

advanced = st.expander("⚙️ Erweiterte Optionen", expanded=False)
with advanced:
    lookback_days = st.slider("Lookback (Tage für S/R & ATR)", 20, 400, 180, step=10)
    news_toggle = st.checkbox(
        "News/Sentiment aus RSS einbeziehen (empfohlen)",
        value=True,
        help="Verwendet kostenlose RSS-Feeds (Google News / Yahoo Finance).",
    )
    risk_reward = st.select_slider(
        "Risikoverhältnis (TP:SL)",
        options=[1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
        value=2.0,
    )
    use_log_chart = st.checkbox("Logarithmische Y-Achse (für Krypto/Indizes sinnvoll)", value=True)

run = st.button("🔎 Analyse starten")

# ---------- Hilfsfunktionen (Technische Indikatoren) ----------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(method="bfill")

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k=14, d=3):
    lowest_low = low.rolling(k).min()
    highest_high = high.rolling(k).max()
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(d).mean()
    return k_percent, d_percent

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_sr(df: pd.DataFrame, window: int = 20):
    """Einfache S/R-Erkennung: lokale Extrema + Perzentile."""
    lows = df['Low'].rolling(window, center=True).min()
    highs = df['High'].rolling(window, center=True).max()
    sr_levels = []
    for idx in df.index:
        if not np.isnan(lows.loc[idx]) and df['Low'].loc[idx] == lows.loc[idx]:
            sr_levels.append(('S', df['Low'].loc[idx]))
        if not np.isnan(highs.loc[idx]) and df['High'].loc[idx] == highs.loc[idx]:
            sr_levels.append(('R', df['High'].loc[idx]))
    # Fallback über Quantile:
    q = df['Close'].quantile([0.1, 0.2, 0.5, 0.8, 0.9]).values.tolist()
    sr_levels += [('S', q[0]), ('S', q[1]), ('R', q[3]), ('R', q[4])]
    # dedupliziere nahe Levels
    sr_levels = sorted(sr_levels, key=lambda x: x[1])
    dedup = []
    tol = df['Close'].iloc[-1] * 0.01  # 1%
    for t, level in sr_levels:
        if not dedup or abs(level - dedup[-1][1]) > tol:
            dedup.append((t, float(level)))
    return dedup

# ---------- News / Fundamentale Kurz-Analyse ----------

POS_WORDS = {"beats", "beat", "surge", "growth", "bull", "upgrade", "record", "profit", "tops", "strong",
             "recover", "rebound", "rally", "accelerate", "optimistic", "buy", "outperform", "approval"}
NEG_WORDS = {"miss", "cuts", "cut", "plunge", "fall", "bear", "downgrade", "loss", "lawsuit", "fraud",
             "weak", "warning", "recall", "delay", "sell", "underperform", "bankrupt", "risk", "decline"}

def fetch_news(symbol: str, limit: int = 12):
    """
    Kostenlose Feeds:
      - Google News (nach Symbol/Name)
      - Yahoo Finance RSS, sofern vorhanden
    """
    feeds = [
        f"https://news.google.com/rss/search?q={symbol}+stock+OR+crypto&hl=en-US&gl=US&ceid=US:en",
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    ]
    items = []
    for url in feeds:
        try:
            d = feedparser.parse(url)
            for e in d.entries[:limit]:
                title = e.title
                link = getattr(e, "link", "")
                published = getattr(e, "published", "") or getattr(e, "updated", "")
                items.append({"title": title, "link": link, "published": published})
        except Exception:
            continue
    # einfache Sentimentzählung
    for it in items:
        t = it["title"].lower()
        pos = sum(1 for w in POS_WORDS if re.search(rf"\\b{re.escape(w)}\\b", t))
        neg = sum(1 for w in NEG_WORDS if re.search(rf"\\b{re.escape(w)}\\b", t))
        it["sentiment"] = pos - neg
    # Score normalisieren
    total = sum([it["sentiment"] for it in items]) if items else 0
    sentiment_score = 0.0
    if items:
        sentiment_score = np.clip(total / (len(items) * 2.0), -1.0, 1.0)  # ~[-1, 1]
    return items[:limit], float(sentiment_score)

# ---------- yfinance Mapping ----------

def tf_to_period_interval(tf: str):
    if tf == "1h":
        return ("60d", "1h")
    if tf == "4h":
        return ("400d", "4h")
    if tf == "1d":
        return ("5y", "1d")
    if tf == "1W":
        return ("10y", "1wk")
    if tf == "1M":
        return ("max", "1mo")
    return ("1y", "1d")

@st.cache_data(show_spinner=False, ttl=600)
def load_data(symbol: str, tf: str):
    period, interval = tf_to_period_interval(tf)
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    df.index = pd.to_datetime(df.index)
    return df

def score_and_recommend(df: pd.DataFrame, news_score: float):
    """Erzeuge eine klare Empfehlung basierend auf einfachen, transparenten Regeln."""
    last = df.iloc[-1]
    close = float(last["Close"])
    # Indikatoren
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    df["EMA21"] = ema(df["Close"], 21)
    rsi_series = rsi(df["Close"], 14)
    rsi_val = float(rsi_series.iloc[-1])
    macd_line, sig_line, hist = macd(df["Close"])
    macd_hist = float(hist.iloc[-1])
    macd_trend = float(macd_line.iloc[-1] - sig_line.iloc[-1])
    k, d = stochastic(df["High"], df["Low"], df["Close"], 14, 3)
    stoch_k, stoch_d = float(k.iloc[-1]), float(d.iloc[-1])
    atr14 = float(atr(df["High"], df["Low"], df["Close"], 14).iloc[-1])

    conditions = []
    score = 0

    # Trend: über/unter gleitenden Durchschnitten
    if close > df["SMA200"].iloc[-1]:
        score += 1; conditions.append("Trend: Kurs > SMA200 (bullisch)")
    else:
        score -= 1; conditions.append("Trend: Kurs < SMA200 (bärisch)")

    if close > df["SMA50"].iloc[-1]:
        score += 0.5; conditions.append("Momentum: Kurs > SMA50")
    else:
        score -= 0.5; conditions.append("Momentum: Kurs < SMA50")

    # RSI
    if rsi_val < 30: score += 0.7; conditions.append(f"RSI {rsi_val:.1f} (überverkauft)")
    elif rsi_val > 70: score -= 0.7; conditions.append(f"RSI {rsi_val:.1f} (überkauft)")
    else: conditions.append(f"RSI {rsi_val:.1f} (neutral)")

    # MACD
    if macd_hist > 0 and macd_trend > 0: score += 0.7; conditions.append("MACD > Signal & Histogramm steigend (bullisch)")
    elif macd_hist < 0 and macd_trend < 0: score -= 0.7; conditions.append("MACD < Signal & Histogramm fallend (bärisch)")
    else: conditions.append("MACD gemischt/neutral")

    # Stochastik
    if stoch_k > stoch_d and stoch_k < 80: score += 0.3; conditions.append("Stochastik: bullisches Kreuz")
    elif stoch_k < stoch_d and stoch_k > 20: score -= 0.3; conditions.append("Stochastik: bärisches Kreuz")
    else: conditions.append("Stochastik neutral/extrem")

    # News / Sentiment
    if not math.isnan(news_score):
        score += 0.5 * news_score
        conditions.append(f"News-Sentiment: {news_score:+.2f}")

    # Empfehlung
    if score >= 1.2:
        rec = "KAUFEN"
    elif score <= -1.2:
        rec = "VERKAUFEN"
    else:
        rec = "HALTEN"

    # Stop-Loss / Take-Profit: ATR-basiert + S/R-Niveau
    sr = detect_sr(df.tail(int(max(lookback_days, 60))))
    supports = [lvl for t, lvl in sr if t == 'S' and lvl < close]
    resistances = [lvl for t, lvl in sr if t == 'R' and lvl > close]
    nearest_support = max(supports) if supports else close - 2*atr14
    nearest_resistance = min(resistances) if resistances else close + 2*atr14

    # ATR fallback, falls NaN
    if np.isnan(atr14) or atr14 <= 0:
        atr14 = max(1e-9, np.std(df["Close"].pct_change().dropna()) * close)

    if rec in ("KAUFEN", "HALTEN"):
        sl = min(nearest_support, close - 1.5*atr14)
        tp = close + (risk_reward * (close - sl))
    else:  # VERKAUFEN (Short)
        sl = max(nearest_resistance, close + 1.5*atr14)
        tp = close - (risk_reward * (sl - close))

    out = {
        "close": close,
        "rsi": rsi_val,
        "macd_hist": macd_hist,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "atr": atr14,
        "sma50": float(df["SMA50"].iloc[-1]),
        "sma200": float(df["SMA200"].iloc[-1]),
        "score": float(score),
        "recommendation": rec,
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "sr_levels": sr,
    }
    return out

def plot_chart(df: pd.DataFrame, use_log: bool, sr_levels, sl=None, tp=None):
    df = df.copy()
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.03)

    # Candles
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                       low=df["Low"], close=df["Close"], name="Kurs"),
        row=1, col=1
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"), row=1, col=1)

    # Volumen (falls vorhanden)
    if "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volumen", opacity=0.4),
            row=2, col=1
        )

    # S/R als horizontale Linien
    for t, lvl in sr_levels:
        fig.add_hline(y=lvl, line_width=1, line_dash="dot",
                      annotation_text=f"{t} {lvl:.2f}", annotation_position="right")

    # SL/TP
    if sl is not None:
        fig.add_hline(y=sl, line_color="red", line_width=2,
                      annotation_text=f"SL {sl:.2f}", annotation_position="left")
    if tp is not None:
        fig.add_hline(y=tp, line_color="green", line_width=2,
                      annotation_text=f"TP {tp:.2f}", annotation_position="left")

    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=560,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(type="log" if use_log else "linear", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ---------- Lauflogik ----------
if run:
    symbol = ticker_input.strip()
    if not symbol:
        st.error("Bitte ein gültiges Symbol eingeben.")
        st.stop()

    with st.spinner("Lade Kursdaten …"):
        df = load_data(symbol, timeframe)

    if df.empty:
        st.error("Keine Daten gefunden. Prüfe das Symbol und den Zeitrahmen.")
        st.stop()

    # optional: News laden
    items, news_score = ([], 0.0)
    if news_toggle:
        with st.spinner("Lese Nachrichten …"):
            items, news_score = fetch_news(symbol, limit=12)

    # Analyse
    result = score_and_recommend(df, news_score)

    # Ausgabe (strukturierte Kacheln)
    st.subheader(f"Ergebnis für **{symbol}** – {timeframe}")
    met1, met2, met3 = st.columns(3)
    met1.metric("Letzter Schluss", f"{result['close']:.4f}")
    met2.metric("Score (−∞…+∞)", f"{result['score']:+.2f}")
    met3.metric("Empfehlung", result["recommendation"])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("RSI(14)", f"{result['rsi']:.1f}")
    kpi2.metric("SMA50", f"{result['sma50']:.2f}")
    kpi3.metric("SMA200", f"{result['sma200']:.2f}")
    kpi4.metric("ATR(14)", f"{result['atr']:.4f}")

    # Chart mit S/R, SL, TP
    plot_chart(
        df.tail(max(300, int(lookback_days))),  # Chart begrenzen für mobile Performance
        use_log_chart,
        result["sr_levels"],
        sl=result["stop_loss"],
        tp=result["take_profit"],
    )

    # Handlungsvorschlag
    st.markdown("### 📌 Handlungsvorschlag")
    colx, coly = st.columns(2)
    with colx:
        st.success(f"**Empfehlung:** {result['recommendation']}")
        st.write(
            f"**Stop-Loss:** `{result['stop_loss']:.4f}`  \n"
            f"**Take-Profit:** `{result['take_profit']:.4f}`  \n"
            f"**Risikoverhältnis (TP:SL):** `{risk_reward:.1f}:1`"
        )
    with coly:
        st.write("**Begründung (Kurzfassung):**")
        # gleiche Logik wie im Score – kurz erläutert
        explain = []
        if result['close'] > result['sma200']:
            explain.append("Aufwärtstrend über SMA200")
        else:
            explain.append("Abwärtstrend unter SMA200")
        if result['rsi'] < 30:
            explain.append("RSI überverkauft → Rebound-Potenzial")
        elif result['rsi'] > 70:
            explain.append("RSI überkauft → Korrektur-Risiko")
        if result['macd_hist'] > 0:
            explain.append("MACD-Momentum positiv")
        else:
            explain.append("MACD-Momentum negativ")
        st.write("• " + "  \n• ".join(explain))

    # Detail: News
    if news_toggle:
        st.markdown("### 🗞️ Nachrichten & Sentiment")
        st.write(f"Aggregierter Sentiment-Score: **{news_score:+.2f}** (-1 bärisch … +1 bullisch)")
        if items:
            for it in items:
                ts = it["published"][:16] if it["published"] else ""
                st.markdown(f"- [{it['title']}]({it['link']})  \n  <small>{ts} • Sentiment: {it['sentiment']:+d}</small>", unsafe_allow_html=True)
        else:
            st.caption("Keine Artikel gefunden (oder Feed blockiert).")

    # Haftungsausschluss
    st.markdown("---")
    st.caption(
        "Dies ist **keine Anlageberatung**. Ergebnisse sind heuristisch und können falsch sein. "
        "Nutze eigenes Risikomanagement."
    )
else:
    st.info("Gib ein Symbol ein, wähle den Zeitrahmen und starte die Analyse.")
