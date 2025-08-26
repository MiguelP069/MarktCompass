# 📊 Trading-Analyse (Streamlit, Single-File)

Eine **einzige** `app.py` für Trading-Analysen (Aktien, Krypto, FX, Indizes).
- Daten: **yfinance**
- Indikatoren: SMA/EMA, **RSI**, **MACD**, **Stochastik**, **ATR** (Pandas-basiert, kein TA-Lib nötig)
- Unterstützungs-/Widerstands-Erkennung (lokale Extrema + Quantile)
- **News-Check** via **RSS** (Google News & Yahoo Finance) + einfache Schlagwort-Sentimentanalyse
- Klare **Kauf/Halten/Verkaufen**-Empfehlung + **Stop-Loss**/**Take-Profit**

## 🚀 Schnellstart (lokal)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
