"""
Ernestbaldeh Scanner
A Streamlit research scanner for Ghana Stock Exchange equities only.

Data note:
- The app attempts Yahoo Finance symbols using the .GH suffix.
- GSE coverage on third-party feeds may be incomplete or stale.
- For reliable historical candles, use a licensed/verified GSE data provider or upload a CSV
  with Date, Open, High, Low, Close, Volume columns and a Symbol column.
- This is a research tool, not investment advice.
"""

from datetime import datetime, timezone
from io import BytesIO
import time

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

APP_NAME = "Ernestbaldeh Scanner"
st.set_page_config(page_title=APP_NAME, page_icon="🇬🇭", layout="wide")

# Source: Ghana Stock Exchange listed-company directory, checked during preparation.
# Security codes can change; refresh this list against the official GSE directory periodically.
GSE_STOCKS = {
    "ACCESS": ("Access Bank Ghana Plc", "Banking"),
    "ADB": ("Agricultural Development Bank Plc", "Banking"),
    "AGA": ("AngloGold Ashanti Plc", "Mining"),
    "AADS": ("AngloGold Ashanti Depositary Shares", "Mining"),
    "ALW": ("Aluworks Ltd", "Manufacturing"),
    "ALLGH": ("Atlantic Lithium Ltd", "Mining"),
    "ASG": ("Asante Gold Corporation", "Mining"),
    "BOPP": ("Benso Oil Palm Plantation Ltd", "Agriculture"),
    "CAL": ("CalBank Plc", "Banking"),
    "CLYD": ("Clydestone Ghana Ltd", "Technology"),
    "CMLT": ("Camelot Ghana Ltd", "Manufacturing"),
    "CPC": ("Cocoa Processing Company Ltd", "Food & Beverage"),
    "DASPHARMA": ("Dannex Ayrton Starwin Plc", "Healthcare"),
    "DIGICUT": ("Digicut Advertising and Production Ltd", "Technology"),
    "EGH": ("Ecobank Ghana Plc", "Banking"),
    "EGL": ("Enterprise Group Plc", "Insurance"),
    "ETI": ("Ecobank Transnational Incorporated", "Banking"),
    "FAB": ("First Atlantic Bank Plc", "Banking"),
    "FML": ("Fan Milk Ltd", "Food & Beverage"),
    "GCB": ("GCB Bank Plc", "Banking"),
    "GGBL": ("Guinness Ghana Breweries Plc", "Food & Beverage"),
    "GLD": ("NewGold Issuer Ltd ETF", "ETF"),
    "GOIL": ("GOIL Plc", "Energy"),
    "HORDS": ("Hords Ltd", "Manufacturing"),
    "IIL": ("Intravenous Infusions Plc", "Healthcare"),
    "KASA": ("Kasapreko Plc", "Food & Beverage"),
    "MAC": ("Mega African Capital Ltd", "Distribution"),
    "MMH": ("Meridian-Marshalls Holdings", "Distribution"),
    "MTNGH": ("Scancom Plc (MTN Ghana)", "Telecommunications"),
    "PBC": ("Produce Buying Company Ltd", "Agriculture"),
    "RBGH": ("Republic Bank Ghana Plc", "Banking"),
    "SAMBA": ("Samba Foods Ltd", "Food & Beverage"),
    "SCB": ("Standard Chartered Bank Ghana Plc", "Banking"),
    "SCB-PREF": ("Standard Chartered Bank Ghana Plc Preference Shares", "Banking"),
    "SIC": ("SIC Insurance Company Ltd", "Insurance"),
    "SOGEGH": ("Société Générale Ghana Ltd", "Banking"),
    "SWL": ("Sam Wood Ltd", "Manufacturing"),
    "TBL": ("Trust Bank Ltd (The Gambia)", "Banking"),
    "TLW": ("Tullow Oil Plc", "Oil & Gas"),
    "TOTAL": ("TotalEnergies Marketing Ghana Plc", "Energy"),
    "UNIL": ("Unilever Ghana Plc", "Consumer Goods"),
    "ZEN": ("ZEN Petroleum Holdings Plc", "Energy"),
}

GREEN = "#16803c"
RED = "#c62828"
YELLOW = "#a66b00"

st.title("🇬🇭 Ernestbaldeh Scanner")
st.caption("Ghana Stock Exchange equities only • SuperTrend + ADX • Momentum • Price-action ranking")
st.info(
    "Data transparency: GSE historical OHLCV coverage can be limited on free feeds. "
    "The scanner never fabricates missing prices. Check each symbol's data date before acting."
)

with st.sidebar:
    st.header("Scanner settings")
    data_source = st.radio("Historical data source", ["Yahoo Finance (.GH)", "Upload GSE OHLCV CSV"])
    fast_atr = st.number_input("Fast SuperTrend ATR period", min_value=2, max_value=50, value=10)
    fast_mult = st.number_input("Fast SuperTrend multiplier", min_value=0.5, max_value=10.0, value=2.5, step=0.1)
    slow_atr = st.number_input("Slow SuperTrend ATR period", min_value=2, max_value=80, value=22)
    slow_mult = st.number_input("Slow SuperTrend multiplier", min_value=0.5, max_value=10.0, value=3.5, step=0.1)
    adx_period = st.number_input("ADX period", min_value=2, max_value=50, value=14)
    adx_strong = st.number_input("Strong trend ADX level", min_value=20, max_value=60, value=25)
    adx_min = st.number_input("Minimum ADX for directional short-term signals", min_value=10, max_value=40, value=20)
    lookback_days = st.selectbox("History to request", [180, 365, 730, 1095], index=2, format_func=lambda x: f"{x} calendar days")
    run_scan = st.button("🔄 Run / Refresh Scan", type="primary", use_container_width=True)

@st.cache_data(ttl=900, show_spinner=False)
def download_yahoo(symbol, period_days):
    if yf is None:
        return pd.DataFrame(), "yfinance is not installed"
    # Yahoo suffix for Ghana listings is not guaranteed for every security.
    candidates = [f"{symbol}.GH"]
    if symbol == "SCB-PREF":
        candidates = ["SCBPREF.GH", "SCB-PREF.GH"]
    for ticker in candidates:
        try:
            df = yf.download(
                ticker,
                period=f"{int(period_days)}d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                required = ["Open", "High", "Low", "Close"]
                if all(c in df.columns for c in required):
                    keep = required + (["Volume"] if "Volume" in df.columns else [])
                    df = df[keep].copy()
                    if "Volume" not in df:
                        df["Volume"] = np.nan
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df.dropna(subset=["High", "Low", "Close"])
                    return df, f"Yahoo Finance: {ticker}"
        except Exception:
            pass
    return pd.DataFrame(), "No usable Yahoo Finance history for this symbol"

def true_range(df):
    prev_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

def atr_series(df, period):
    return true_range(df).ewm(alpha=1 / int(period), adjust=False, min_periods=int(period)).mean()

def supertrend(df, period, multiplier):
    atr = atr_series(df, period)
    mid = (df["High"] + df["Low"]) / 2
    upper = mid + float(multiplier) * atr
    lower = mid - float(multiplier) * atr
    final_upper = upper.copy()
    final_lower = lower.copy()
    direction = pd.Series(index=df.index, dtype="float64")
    line = pd.Series(index=df.index, dtype="float64")
    for i in range(len(df)):
        if i == 0 or pd.isna(atr.iloc[i]):
            direction.iloc[i] = np.nan
            line.iloc[i] = np.nan
            continue
        if i == 1 or pd.isna(final_upper.iloc[i-1]):
            final_upper.iloc[i] = upper.iloc[i]
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_upper.iloc[i] = (
                min(upper.iloc[i], final_upper.iloc[i-1])
                if df["Close"].iloc[i-1] <= final_upper.iloc[i-1] else upper.iloc[i]
            )
            final_lower.iloc[i] = (
                max(lower.iloc[i], final_lower.iloc[i-1])
                if df["Close"].iloc[i-1] >= final_lower.iloc[i-1] else lower.iloc[i]
            )
        if i <= 1 or pd.isna(direction.iloc[i-1]):
            direction.iloc[i] = 1 if df["Close"].iloc[i] >= mid.iloc[i] else -1
        elif direction.iloc[i-1] < 0:
            direction.iloc[i] = 1 if df["Close"].iloc[i] > final_upper.iloc[i] else -1
        else:
            direction.iloc[i] = -1 if df["Close"].iloc[i] < final_lower.iloc[i] else 1
        line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] > 0 else final_upper.iloc[i]
    return direction, line

def adx_series(df, period):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    atr = atr_series(df, period).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di

def analyze_symbol(symbol, company, sector, df, source, cfg):
    if df is None or df.empty or len(df) < max(cfg["slow_atr"], cfg["adx_period"]) * 3:
        return None
    df = df.sort_index().copy()
    df = df[~df.index.duplicated(keep="last")]
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < max(cfg["slow_atr"], cfg["adx_period"]) * 3:
        return None

    fast_dir, fast_line = supertrend(df, cfg["fast_atr"], cfg["fast_mult"])
    slow_dir, slow_line = supertrend(df, cfg["slow_atr"], cfg["slow_mult"])
    adx, plus_di, minus_di = adx_series(df, cfg["adx_period"])
    close = df["Close"]
    latest = close.iloc[-1]
    adx_now = adx.iloc[-1]
    pdi = plus_di.iloc[-1]
    mdi = minus_di.iloc[-1]
    fd = fast_dir.iloc[-1]
    sd = slow_dir.iloc[-1]
    mom5 = (latest / close.iloc[-6] - 1) * 100 if len(close) >= 6 and close.iloc[-6] else np.nan
    mom20 = (latest / close.iloc[-21] - 1) * 100 if len(close) >= 21 and close.iloc[-21] else np.nan
    recent_confirm = (close.iloc[-1] > close.iloc[-2]) if len(close) >= 2 else False
    price_above_fast = bool(latest > fast_line.iloc[-1]) if pd.notna(fast_line.iloc[-1]) else False
    price_above_slow = bool(latest > slow_line.iloc[-1]) if pd.notna(slow_line.iloc[-1]) else False

    # Classification prioritizes agreement; ADX alone never defines sideways.
    directional = pd.notna(adx_now) and pd.notna(pdi) and pd.notna(mdi)
    if directional and sd > 0 and fd > 0 and pdi > mdi and adx_now >= cfg["adx_min"]:
        category = "Long-Term Bullish"
        signal = "BULLISH"
    elif directional and sd < 0 and fd < 0 and mdi > pdi and adx_now >= cfg["adx_min"]:
        category = "Long-Term Bearish"
        signal = "BEARISH"
    elif directional and fd > 0 and pdi > mdi and adx_now >= cfg["adx_min"]:
        category = "Short-Term Bullish"
        signal = "BULLISH"
    elif directional and fd < 0 and mdi > pdi and adx_now >= cfg["adx_min"]:
        category = "Short-Term Bearish"
        signal = "BEARISH"
    elif (pd.notna(adx_now) and adx_now < cfg["adx_min"]) or (fd != sd) or not directional:
        category = "Sideways Market"
        signal = "MIXED"
    else:
        # Conflicting directional confirmation is not forced into a trade category.
        category = "Sideways Market"
        signal = "MIXED"

    agreement = 2 if fd == sd else 0
    strength = min(float(adx_now), 50) / 50 if pd.notna(adx_now) else 0
    momentum_score = np.nan_to_num(mom5, nan=0.0) * 0.35 + np.nan_to_num(mom20, nan=0.0) * 0.65
    directional_bonus = 1 if ((signal == "BULLISH" and pdi > mdi) or (signal == "BEARISH" and mdi > pdi)) else 0
    score = agreement + strength + directional_bonus + np.sign(momentum_score) * min(abs(momentum_score) / 100, 1)
    return {
        "Category": category, "Rank Score": round(score, 4), "Company": company, "Ticker": symbol,
        "Sector": sector, "Latest Price (GHS)": round(float(latest), 4),
        "Fast ST": "Bullish" if fd > 0 else "Bearish",
        "Slow ST": "Bullish" if sd > 0 else "Bearish",
        "ADX (14)": round(float(adx_now), 2) if pd.notna(adx_now) else np.nan,
        "+DI": round(float(pdi), 2) if pd.notna(pdi) else np.nan,
        "−DI": round(float(mdi), 2) if pd.notna(mdi) else np.nan,
        "5D Momentum %": round(float(mom5), 2) if pd.notna(mom5) else np.nan,
        "20D Momentum %": round(float(mom20), 2) if pd.notna(mom20) else np.nan,
        "Latest Data Date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        "Recent Price Action": "Up day" if recent_confirm else "Flat/down day",
        "Data Source": source,
    }

def load_csv(uploaded):
    try:
        df = pd.read_csv(uploaded)
        df.columns = [str(c).strip() for c in df.columns]
        lower = {c.lower(): c for c in df.columns}
        aliases = {
            "date": ["date", "datetime", "timestamp"],
            "symbol": ["symbol", "ticker", "share code", "share_code"],
            "open": ["open", "opening price"],
            "high": ["high", "day high"],
            "low": ["low", "day low"],
            "close": ["close", "closing price", "closing price - vwap", "price"],
            "volume": ["volume", "shares traded", "total shares traded"],
        }
        found = {}
        for target, options in aliases.items():
            for option in options:
                if option in lower:
                    found[target] = lower[option]
                    break
        required = ["date", "symbol", "open", "high", "low", "close"]
        if any(x not in found for x in required):
            st.error("CSV needs columns for Date, Symbol, Open, High, Low and Close. Volume is optional.")
            return None
        out = pd.DataFrame()
        for target, source_col in found.items():
            out[target.title()] = df[source_col]
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
        for c in ["Open", "High", "Low", "Close"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        if "Volume" not in out:
            out["Volume"] = np.nan
        out = out.dropna(subset=["Date", "Symbol", "High", "Low", "Close"])
        out = out[out["Symbol"].isin(GSE_STOCKS.keys())]
        return out
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return None

cfg = {
    "fast_atr": int(fast_atr), "fast_mult": float(fast_mult),
    "slow_atr": int(slow_atr), "slow_mult": float(slow_mult),
    "adx_period": int(adx_period), "adx_min": float(adx_min),
}

uploaded_csv = None
if data_source == "Upload GSE OHLCV CSV":
    uploaded_csv = st.file_uploader("Upload daily GSE historical prices (CSV)", type=["csv"])
    st.caption("Use one row per trading day per share code. Only symbols in this GSE universe are scanned.")

if "scan_data" not in st.session_state:
    st.session_state["scan_data"] = None
if run_scan:
    all_rows, statuses = [], []
    progress = st.progress(0)
    status_text = st.empty()
    if data_source == "Upload GSE OHLCV CSV":
        if uploaded_csv is None:
            st.warning("Upload a CSV first, then press Run / Refresh Scan.")
            st.stop()
        csv_data = load_csv(uploaded_csv)
        if csv_data is None or csv_data.empty:
            st.warning("No valid rows for recognized GSE symbols were found in that CSV.")
            st.stop()
        groups = list(GSE_STOCKS.items())
        for i, (symbol, (company, sector)) in enumerate(groups):
            hist = csv_data[csv_data["Symbol"] == symbol].set_index("Date").sort_index()
            if not hist.empty:
                hist = hist[["Open", "High", "Low", "Close", "Volume"]]
                row = analyze_symbol(symbol, company, sector, hist, "Uploaded GSE CSV", cfg)
                if row:
                    all_rows.append(row)
                statuses.append({"Ticker": symbol, "Company": company, "Status": "Loaded" if row else "Insufficient history"})
            progress.progress((i + 1) / len(groups))
            status_text.write(f"Processing {symbol}…")
    else:
        groups = list(GSE_STOCKS.items())
        for i, (symbol, (company, sector)) in enumerate(groups):
            hist, source = download_yahoo(symbol, lookback_days)
            if not hist.empty:
                row = analyze_symbol(symbol, company, sector, hist, source, cfg)
                if row:
                    all_rows.append(row)
                    statuses.append({"Ticker": symbol, "Company": company, "Status": "Scanned", "Data Date": row["Latest Data Date"], "Source": source})
                else:
                    statuses.append({"Ticker": symbol, "Company": company, "Status": "Insufficient history", "Source": source})
            else:
                statuses.append({"Ticker": symbol, "Company": company, "Status": "No data", "Source": source})
            progress.progress((i + 1) / len(groups))
            status_text.write(f"Checking {symbol} ({i+1}/{len(groups)})…")
            time.sleep(0.02)
    progress.empty()
    status_text.empty()
    st.session_state["scan_data"] = {
        "results": pd.DataFrame(all_rows),
        "statuses": pd.DataFrame(statuses),
        "scanned_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

data = st.session_state.get("scan_data")
if data is None:
    st.subheader("How to use")
    st.markdown(
        "1. Choose **Yahoo Finance (.GH)** for an automatic attempt at third-party historical prices, "
        "or upload a GSE OHLCV CSV for the symbols you have data for.\n"
        "2. Set the indicator periods in the sidebar.\n"
        "3. Press **Run / Refresh Scan**.\n"
        "4. Review category tables, verify each data date, and download the results."
    )
    st.markdown("### Five scan categories")
    st.markdown(
        "🟢 **Long-Term Bullish** · 🟢 **Short-Term Bullish** · "
        "🔴 **Long-Term Bearish** · 🔴 **Short-Term Bearish** · 🟡 **Sideways Market**"
    )
    st.caption("No selections are filled in with guessed prices. The first scan may take a minute because each listed security is checked.")
    st.stop()

results = data["results"].copy()
statuses = data["statuses"].copy()
st.caption(f"Last scan: {data['scanned_at']} • Universe: {len(GSE_STOCKS)} GSE securities")

if results.empty:
    st.error(
        "No symbols produced enough usable historical data for the indicators. This may mean the free Yahoo Finance "
        "feed does not cover GSE symbols. Try uploading a verified GSE historical OHLCV CSV. The scanner will not "
        "substitute US stocks or fabricate Ghana prices."
    )
else:
    scanned_count = statuses["Status"].isin(["Scanned", "Loaded"]).sum() if not statuses.empty else len(results)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks successfully scanned", int(scanned_count))
    c2.metric("Bullish categories", int(results["Category"].isin(["Long-Term Bullish", "Short-Term Bullish"]).sum()))
    c3.metric("Bearish categories", int(results["Category"].isin(["Long-Term Bearish", "Short-Term Bearish"]).sum()))
    c4.metric("Sideways / mixed", int((results["Category"] == "Sideways Market").sum()))

    category_settings = [
        ("Long-Term Bullish", "🟢 Long-Term Bullish", GREEN),
        ("Long-Term Bearish", "🔴 Long-Term Bearish", RED),
        ("Short-Term Bullish", "🟢 Short-Term Bullish", GREEN),
        ("Short-Term Bearish", "🔴 Short-Term Bearish", RED),
        ("Sideways Market", "🟡 Sideways Market", YELLOW),
    ]
    tabs = st.tabs([x[1] for x in category_settings] + ["All Results", "Data Coverage"])
    display_cols = [
        "Company", "Ticker", "Sector", "Latest Price (GHS)", "Fast ST", "Slow ST",
        "ADX (14)", "+DI", "−DI", "5D Momentum %", "20D Momentum %",
        "Recent Price Action", "Latest Data Date"
    ]
    for tab, (category, label, color) in zip(tabs[:5], category_settings):
        with tab:
            subset = results[results["Category"] == category].sort_values("Rank Score", ascending=False).head(5)
            if subset.empty:
                st.warning("No stocks currently qualify for this category. No selections have been invented.")
            else:
                st.caption(f"Showing up to five qualifying stocks. Ranking considers indicator agreement, ADX, momentum and recent price action.")
                st.dataframe(
                    subset[display_cols].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Latest Price (GHS)": st.column_config.NumberColumn(format="GH₵ %.4f"),
                        "ADX (14)": st.column_config.NumberColumn(format="%.2f"),
                        "5D Momentum %": st.column_config.NumberColumn(format="%.2f%%"),
                        "20D Momentum %": st.column_config.NumberColumn(format="%.2f%%"),
                    }
                )
    with tabs[5]:
        st.subheader("All scanned GSE securities")
        view = results.sort_values(["Category", "Rank Score"], ascending=[True, False])
        st.dataframe(view[display_cols + ["Rank Score", "Data Source"]], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full scan CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name=f"ernestbaldeh_gse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    with tabs[6]:
        st.subheader("Data coverage and freshness")
        if statuses.empty:
            st.info("No per-symbol status rows were returned.")
        else:
            st.dataframe(statuses, use_container_width=True, hide_index=True)
            no_data = statuses[statuses["Status"] == "No data"] if "Status" in statuses else pd.DataFrame()
            if not no_data.empty:
                st.warning(f"{len(no_data)} listed securities had no usable data from the selected source.")
        st.markdown("**Important:** an unchanged price does not necessarily mean a security was actively traded. Check trading volume and the latest data date before interpreting a signal.")

st.divider()
st.subheader("Methodology & safeguards")
st.markdown(
    f"""
- **Fast SuperTrend:** ATR {int(fast_atr)}, multiplier {float(fast_mult):.1f}.
- **Slow SuperTrend:** ATR {int(slow_atr)}, multiplier {float(slow_mult):.1f}.
- **ADX:** period {int(adx_period)}; reference levels 20 and {int(adx_strong)}.
- **Long-term signals:** require agreement between fast and slow SuperTrend plus directional confirmation.
- **Short-term signals:** use fast SuperTrend and ADX/directional confirmation.
- **Sideways:** includes weak or conflicting evidence; ADX alone does not determine the category.
- **Ranking:** combines SuperTrend agreement, ADX strength, directional movement, 5-day momentum, 20-day momentum and recent price action.
- **Market scope:** Ghana Stock Exchange securities listed in the embedded universe only; no US, forex, crypto or commodity instruments are scanned.
- **Limitations:** illiquidity, stale prices, thin trading and corporate actions can distort technical indicators. Signals are research outputs, not guarantees or personalized financial advice.
"""
)
st.caption("Official reference: Ghana Stock Exchange listed companies and market-data pages. Verify the current security list and data permissions before production use.")
