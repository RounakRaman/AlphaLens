"""
Market-Neutral Long-Short Factor Strategy — Streamlit App
Converts the original matplotlib backtest into a fully interactive dashboard.
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Long-Short Factor Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Dark-theme CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    html, body, .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px 20px; text-align: center;
    }
    .metric-label { font-size: 12px; color: #8b949e; margin-bottom: 4px; }
    .metric-value { font-size: 22px; font-weight: 700; }
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    .neutral  { color: #58a6ff; }
    .signal-box {
        background: #161b22; border: 2px solid;
        border-radius: 12px; padding: 20px 28px; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 6px;
        color: #8b949e; border: 1px solid #30363d;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important; color: #e6edf3 !important;
        border-color: #58a6ff !important;
    }
    h1, h2, h3 { color: #e6edf3 !important; }
    .stSlider > div > div { background: #30363d; }
    div[data-testid="stMetric"] { background: #161b22; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Colours
# ─────────────────────────────────────────
LC = '#3fb950'   # long / green
SC = '#f85149'   # short / red
BC = '#58a6ff'   # blue
MC = '#d29922'   # market / gold
PC = '#bc8cff'   # purple
GR = '#8b949e'   # grey
OC = '#ffa657'   # orange

# ─────────────────────────────────────────
#  Dummy data generator
# ─────────────────────────────────────────
@st.cache_data
def generate_dummy_data():
    """Generate realistic S&P 500-like OHLCV + RSI daily data (2017-2024)."""
    np.random.seed(42)
    dates = pd.date_range("2017-01-03", "2024-05-31", freq="B")
    n = len(dates)

    price = 2250.0
    prices, highs, lows, opens, volumes = [], [], [], [], []
    for _ in range(n):
        ret = np.random.normal(0.00035, 0.0105)
        price *= (1 + ret)
        daily_range = price * np.random.uniform(0.005, 0.022)
        opens.append(price * (1 + np.random.normal(0, 0.003)))
        highs.append(price + daily_range * np.random.uniform(0.3, 1.0))
        lows.append(price - daily_range * np.random.uniform(0.3, 1.0))
        prices.append(price)
        volumes.append(int(np.random.lognormal(21, 0.4)))

    df = pd.DataFrame({
        "formatted_date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "adjclose": prices,
        "volume": volumes,
    })

    # Compute RSI-14
    delta = df["adjclose"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi"].fillna(50, inplace=True)
    return df

# ─────────────────────────────────────────
#  Core strategy functions
# ─────────────────────────────────────────
def load_csv(uploaded):
    df = pd.read_csv(uploaded, parse_dates=True)
    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.rename(columns={date_col: "formatted_date"}, inplace=True)
    # must have adjclose or close
    if "adjclose" not in df.columns and "close" in df.columns:
        df["adjclose"] = df["close"]
    df.sort_values("formatted_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # add RSI if missing
    if "rsi" not in df.columns:
        delta = df["adjclose"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100 - 100 / (1 + rs)
        df["rsi"].fillna(50, inplace=True)
    return df


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={"formatted_date": "date"}, inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["log_ret"] = np.log(df["adjclose"] / df["adjclose"].shift(1))
    df["prev_close"] = df["adjclose"].shift(1)
    df["TR"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"]  - df["prev_close"]).abs(),
        ),
    )
    df["ym"] = df["date"].dt.to_period("M")
    return df


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("ym").agg(
        date     =("date",     "last"),
        close    =("adjclose", "last"),
        open_m   =("open",     "first"),
        high_m   =("high",     "max"),
        low_m    =("low",      "min"),
        volume_m =("volume",   "sum"),
        rsi_last =("rsi",      "last"),
        rsi_mean =("rsi",      "mean"),
        n_days   =("log_ret",  "count"),
    ).reset_index()
    agg["ret_m"] = agg["close"].pct_change()
    daily_vol = (
        df.groupby("ym")["log_ret"]
        .std()
        .rename("real_vol_m") * np.sqrt(252)
    )
    agg = agg.merge(daily_vol.reset_index(), on="ym", how="left")
    agg.sort_values("date", inplace=True)
    agg.reset_index(drop=True, inplace=True)
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def ts_zscore(s, window=36, min_p=12):
    mu  = s.rolling(window, min_periods=min_p).mean()
    sig = s.rolling(window, min_periods=min_p).std()
    return ((s - mu) / (sig + 1e-9)).clip(-3.0, 3.0)


def build_factors(m: pd.DataFrame, roll_win: int = 36) -> tuple:
    m = m.copy()
    m["inv_rsi"]      = 100.0 - m["rsi_last"]
    m["sma12"]        = m["close"].rolling(12, min_periods=6).mean()
    m["price_vs_sma"] = (m["sma12"] - m["close"]) / (m["sma12"] + 1e-9)
    m["F_value"]      = (0.5 * ts_zscore(m["inv_rsi"],      roll_win)
                       + 0.5 * ts_zscore(m["price_vs_sma"], roll_win))

    m["mom_12_1"]   = m["close"].shift(1) / m["close"].shift(12) - 1.0
    m["F_momentum"] = ts_zscore(m["mom_12_1"], roll_win)

    avg_vol        = m["volume_m"].rolling(12, min_periods=6).mean()
    m["vol_ratio"] = m["volume_m"] / (avg_vol + 1e-9) - 1.0
    m["vpt"]       = m["ret_m"] * m["vol_ratio"]
    m["F_quality"] = ts_zscore(m["vpt"], roll_win)

    m["F_lowvol"] = ts_zscore(-m["real_vol_m"], roll_win)

    FCOLS = ["F_value", "F_momentum", "F_quality", "F_lowvol"]
    m["composite"] = m[FCOLS].mean(axis=1)

    roll_std        = m["composite"].rolling(roll_win, min_periods=12).std()
    m["signal_raw"] = (m["composite"] / (2.0 * roll_std + 1e-9)).clip(-1.0, 1.0)
    return m, FCOLS


def apply_regime_neutrality(m: pd.DataFrame, rsi_low: float, rsi_high: float,
                             roll_win: int) -> pd.DataFrame:
    m = m.copy()

    def classify(rsi):
        if rsi < rsi_low:   return "Oversold"
        if rsi < rsi_high:  return "Neutral"
        return "Overbought"

    m["regime"] = m["rsi_last"].apply(classify)
    neutral_sigs = []
    for i in range(len(m)):
        start      = max(0, i - roll_win + 1)
        window     = m.iloc[start: i + 1]
        cur_regime = m.iloc[i]["regime"]
        reg_rows   = window[window["regime"] == cur_regime]
        if len(reg_rows) >= 3:
            rm       = reg_rows["signal_raw"].mean()
            rs       = reg_rows["signal_raw"].std()
            demeaned = (m.iloc[i]["signal_raw"] - rm) / (rs + 1e-9)
            neutral_sigs.append(float(np.clip(demeaned, -1.0, 1.0)))
        else:
            neutral_sigs.append(float(m.iloc[i]["signal_raw"]))
    m["signal_neutral"] = neutral_sigs
    return m


def apply_vol_targeting(m: pd.DataFrame, df_daily: pd.DataFrame,
                         target_vol: float, halflife: int, vol_window: int) -> pd.DataFrame:
    m = m.copy()
    ewm_vols = []
    for dt in m["date"]:
        hist = df_daily[df_daily["date"] <= dt].tail(vol_window)
        if len(hist) >= 20:
            ev      = hist["log_ret"].ewm(halflife=halflife, min_periods=15).std().iloc[-1]
            ann_vol = ev * np.sqrt(252)
        else:
            ann_vol = target_vol
        ewm_vols.append(ann_vol)
    m["forecast_vol"]  = ewm_vols
    m["vol_scale"]     = (target_vol / m["forecast_vol"]).clip(0.10, 2.0)
    m["signal_voltgt"] = (m["signal_neutral"] * m["vol_scale"]).clip(-1.0, 1.0)
    return m


def apply_kelly_sizing(m: pd.DataFrame, kelly_frac: float, roll_win: int) -> pd.DataFrame:
    m = m.copy()
    roll_mu    = m["composite"].rolling(roll_win, min_periods=12).mean()
    roll_var   = m["composite"].rolling(roll_win, min_periods=12).var()
    kelly_raw  = roll_mu / (roll_var + 1e-9)
    norm_scale = kelly_raw.abs().rolling(roll_win, min_periods=12).quantile(0.95)
    kelly_norm = kelly_raw / (norm_scale + 1e-9)
    m["signal_kelly"] = (kelly_norm * kelly_frac).clip(-1.0, 1.0)
    return m


def run_backtest(m: pd.DataFrame, sig_col: str, cost_bps: float, name: str) -> pd.DataFrame:
    cost = cost_bps / 10_000.0
    keep = ["date", "ret_m", sig_col, "real_vol_m", "regime"]
    for opt in ("vol_scale", "forecast_vol"):
        if opt in m.columns:
            keep.append(opt)
    bt = m[keep].dropna(subset=[sig_col]).copy()
    bt["sig_lag"]    = bt[sig_col].shift(1)
    bt["pnl_gross"]  = bt["sig_lag"] * bt["ret_m"]
    bt["turnover"]   = bt["sig_lag"].diff().abs()
    bt["tc"]         = bt["turnover"] * cost
    bt["pnl_net"]    = bt["pnl_gross"] - bt["tc"]
    bt["pnl_market"] = bt["ret_m"]
    bt = bt.dropna(subset=["pnl_net"]).copy()
    bt["nav"]        = (1.0 + bt["pnl_net"]).cumprod()
    bt["nav_gross"]  = (1.0 + bt["pnl_gross"]).cumprod()
    bt["nav_market"] = (1.0 + bt["pnl_market"]).cumprod()
    bt["version"]    = name
    bt.reset_index(drop=True, inplace=True)
    return bt


def compute_metrics(rets: pd.Series, freq: int = 12, label: str = "") -> dict:
    r = rets.dropna()
    n = len(r)
    if n < 3:
        return {}
    ann = float((1.0 + r).prod() ** (freq / n) - 1.0)
    vol = float(r.std() * np.sqrt(freq))
    sha = ann / vol if vol > 1e-9 else np.nan
    nav = (1.0 + r).cumprod()
    mdd = float(((nav - nav.cummax()) / nav.cummax()).min())
    dn  = r[r < 0]
    sor = ann / (dn.std() * np.sqrt(freq)) if len(dn) > 1 else np.nan
    cal = ann / abs(mdd) if abs(mdd) > 1e-9 else np.nan
    hit = float((r > 0).mean())
    return dict(label=label, ann_ret=ann, ann_vol=vol, sharpe=sha,
                sortino=sor, mdd=mdd, calmar=cal, hit=hit, n=n)


def signal_to_advice(signal: float, version: str) -> dict:
    """Convert composite signal to human-readable advice."""
    s = float(signal) if not np.isnan(signal) else 0.0
    if s > 0.5:
        action = "STRONG LONG"
        desc   = "High conviction long. All factor signals aligned bullishly."
        color  = LC
        icon   = "🟢"
    elif s > 0.15:
        action = "LONG"
        desc   = "Moderate long bias. Majority of factors support upside."
        color  = LC
        icon   = "🟩"
    elif s > -0.15:
        action = "NEUTRAL / FLAT"
        desc   = "Mixed factor signals. No strong directional edge."
        color  = GR
        icon   = "⬜"
    elif s > -0.5:
        action = "SHORT"
        desc   = "Moderate short bias. Majority of factors suggest downside."
        color  = SC
        icon   = "🟥"
    else:
        action = "STRONG SHORT"
        desc   = "High conviction short. All factor signals aligned bearishly."
        color  = SC
        icon   = "🔴"
    return {"action": action, "desc": desc, "color": color, "icon": icon, "signal": s}


# ─────────────────────────────────────────
#  Sidebar — Parameters
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Strategy Parameters")

    st.markdown("### 📂 Data Source")
    data_source = st.radio("Data Source", ["Use built-in dummy data", "Upload my CSV"], label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader(
            "Upload OHLCV CSV (needs: date, open, high, low, close/adjclose, volume)",
            type=["csv"]
        )
        st.caption("Optional column: `rsi` — will be computed if missing.")

    st.divider()
    st.markdown("### 🔧 Tuning Parameters")

    cost_bps = st.slider("Transaction Cost (bps)", 0, 50, 10, 1,
                          help="One-way cost per unit of turnover")
    target_vol = st.slider("Vol Target (% annual)", 5, 30, 10, 1,
                            help="EWMA vol targeting level") / 100.0
    kelly_frac = st.slider("Kelly Fraction", 0.1, 1.0, 0.5, 0.05,
                            help="0.5 = half-Kelly sizing")
    rsi_low    = st.slider("RSI Oversold threshold", 20, 50, 40, 1)
    rsi_high   = st.slider("RSI Overbought threshold", 51, 80, 65, 1)
    roll_win   = st.slider("Rolling window (months)", 12, 60, 36, 1,
                            help="Z-score normalisation window")
    vol_halflife = st.slider("EWMA Vol Halflife (days)", 5, 63, 21, 1)
    vol_window   = st.slider("Vol Lookback (days)", 63, 252, 126, 5)

    st.divider()
    run_btn = st.button("▶  Run Strategy", type="primary", width="stretch")

# ─────────────────────────────────────────
#  Header
# ─────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; font-size:28px; margin-bottom:4px;'>
  📈 Market-Neutral Long-Short Factor Strategy
</h1>
<p style='text-align:center; color:#8b949e; font-size:14px; margin-top:0;'>
  Value · Momentum · Quality · Low-Vol  |  Monthly Rebalance  |  Dollar-Neutral
</p>
""", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

if run_btn or st.session_state.results is None:
    with st.spinner("Running strategy pipeline …"):
        # 1. Load data
        if data_source == "Upload my CSV" and uploaded_file:
            raw_df = load_csv(uploaded_file)
        else:
            raw_df = generate_dummy_data()

        df_daily = prepare_daily(raw_df)

        # 2. Monthly resample
        m = resample_monthly(df_daily)

        # 3. Factors
        m, FCOLS = build_factors(m, roll_win)

        # 4. Regime neutrality
        m = apply_regime_neutrality(m, rsi_low, rsi_high, roll_win)

        # 5. Vol targeting
        m = apply_vol_targeting(m, df_daily, target_vol, vol_halflife, vol_window)

        # 6. Kelly sizing
        m = apply_kelly_sizing(m, kelly_frac, roll_win)

        # 7. Backtests
        bt_b = run_backtest(m, "signal_raw",     cost_bps, "Base")
        bt_n = run_backtest(m, "signal_neutral",  cost_bps, "Regime-Neutral")
        bt_v = run_backtest(m, "signal_voltgt",   cost_bps, "Vol-Target")
        bt_k = run_backtest(m, "signal_kelly",    cost_bps, "Kelly")

        # 8. Metrics
        metrics = {
            "Base":           compute_metrics(bt_b["pnl_net"],    label="Base"),
            "Regime-Neutral": compute_metrics(bt_n["pnl_net"],    label="Regime-Neutral"),
            "Vol-Target":     compute_metrics(bt_v["pnl_net"],    label="Vol-Target"),
            "Kelly":          compute_metrics(bt_k["pnl_net"],    label="Kelly"),
            "Market":         compute_metrics(bt_b["pnl_market"], label="Market"),
        }

        # 9. Latest signal advice
        last_row   = m.dropna(subset=["signal_voltgt"]).iloc[-1]
        adv_base   = signal_to_advice(last_row["signal_raw"],    "Base")
        adv_vt     = signal_to_advice(last_row["signal_voltgt"], "Vol-Target")
        adv_kelly  = signal_to_advice(last_row.get("signal_kelly", np.nan), "Kelly")
        composite  = float(last_row["composite"])

        st.session_state.results = dict(
            m=m, df_daily=df_daily, FCOLS=FCOLS,
            bt_b=bt_b, bt_n=bt_n, bt_v=bt_v, bt_k=bt_k,
            metrics=metrics,
            adv_base=adv_base, adv_vt=adv_vt, adv_kelly=adv_kelly,
            composite=composite, last_row=last_row,
        )

res = st.session_state.results
m      = res["m"]
FCOLS  = res["FCOLS"]
bt_b   = res["bt_b"]
bt_n   = res["bt_n"]
bt_v   = res["bt_v"]
bt_k   = res["bt_k"]
metrics = res["metrics"]

# ─────────────────────────────────────────
#  SECTION 1: Signal / Advice Banner
# ─────────────────────────────────────────
st.markdown("## 🎯 Current Signal & Advice")

adv_vt    = res["adv_vt"]
composite = res["composite"]
last_row  = res["last_row"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    color_class = "positive" if composite > 0 else "negative" if composite < 0 else "neutral"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Composite Alpha Score</div>
        <div class="metric-value {color_class}">{composite:+.3f}</div>
        <div style='color:#8b949e; font-size:11px; margin-top:6px;'>Range: −1 to +1</div>
    </div>""", unsafe_allow_html=True)

with col2:
    sig_vt = float(last_row.get("signal_voltgt", 0))
    c2 = "positive" if sig_vt > 0 else "negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Vol-Targeted Signal</div>
        <div class="metric-value {c2}">{sig_vt:+.3f}</div>
        <div style='color:#8b949e; font-size:11px; margin-top:6px;'>After vol scaling</div>
    </div>""", unsafe_allow_html=True)

with col3:
    sig_k = float(last_row.get("signal_kelly", 0))
    c3 = "positive" if sig_k > 0 else "negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Kelly Signal</div>
        <div class="metric-value {c3}">{sig_k:+.3f}</div>
        <div style='color:#8b949e; font-size:11px; margin-top:6px;'>Half-Kelly sized</div>
    </div>""", unsafe_allow_html=True)

with col4:
    regime = str(last_row.get("regime", "N/A"))
    r_color = "positive" if regime == "Oversold" else "negative" if regime == "Overbought" else "neutral"
    rsi_val = float(last_row.get("rsi_last", 50))
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RSI Regime</div>
        <div class="metric-value {r_color}">{regime}</div>
        <div style='color:#8b949e; font-size:11px; margin-top:6px;'>RSI = {rsi_val:.1f}</div>
    </div>""", unsafe_allow_html=True)

# Advice box
st.markdown("<br>", unsafe_allow_html=True)
border_color = adv_vt["color"]
st.markdown(f"""
<div class="signal-box" style="border-color:{border_color}; background: #161b22;">
    <div style='font-size:32px; margin-bottom:6px;'>{adv_vt["icon"]}</div>
    <div style='font-size:24px; font-weight:800; color:{border_color};'>{adv_vt["action"]}</div>
    <div style='color:#c9d1d9; font-size:14px; margin-top:8px;'>{adv_vt["desc"]}</div>
    <div style='color:#8b949e; font-size:12px; margin-top:8px;'>
        Based on Vol-Targeted signal: <b style='color:{border_color};'>{adv_vt["signal"]:+.3f}</b>
        &nbsp;·&nbsp; As of: <b>{str(last_row["date"])[:10]}</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────
#  SECTION 2: KPI Table
# ─────────────────────────────────────────
st.markdown("## 📊 Performance KPIs")

kpi_rows = []
for version in ["Base", "Regime-Neutral", "Vol-Target", "Kelly", "Market"]:
    kpi = metrics[version]
    if not kpi:
        continue
    kpi_rows.append({
        "Version":    version,
        "Ann Return": f"{kpi['ann_ret']*100:+.1f}%",
        "Ann Vol":    f"{kpi['ann_vol']*100:.1f}%",
        "Sharpe":     f"{kpi['sharpe']:.3f}",
        "Sortino":    f"{kpi['sortino']:.3f}",
        "Max DD":     f"{kpi['mdd']*100:.1f}%",
        "Calmar":     f"{kpi['calmar']:.3f}",
        "Hit Rate":   f"{kpi['hit']*100:.1f}%",
    })

kpi_df = pd.DataFrame(kpi_rows)
st.dataframe(
    kpi_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Version":    st.column_config.TextColumn("Version", width="medium"),
        "Ann Return": st.column_config.TextColumn("Ann Return"),
        "Sharpe":     st.column_config.TextColumn("Sharpe"),
    }
)

st.divider()

# ─────────────────────────────────────────
#  SECTION 3: Charts (tabbed)
# ─────────────────────────────────────────
st.markdown("## 📉 Strategy Analytics")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Equity Curves",
    "🌡️ Monthly Returns",
    "🔵 Factor Signals",
    "⚡ Volatility",
    "📐 Factor Correlations",
])

# ---------- Tab 1: Equity Curves ----------
with tab1:
    fig = go.Figure()
    for bt, col, name, dash in [
        (bt_b, BC, "Base",           "solid"),
        (bt_n, LC, "Regime-Neutral", "dash"),
        (bt_v, PC, "Vol-Target 10%", "dot"),
        (bt_k, OC, "Kelly Sizing",   "dashdot"),
    ]:
        fig.add_trace(go.Scatter(
            x=bt["date"], y=bt["nav"],
            name=name, line=dict(color=col, width=2, dash=dash),
        ))
    fig.add_trace(go.Scatter(
        x=bt_b["date"], y=bt_b["nav_market"],
        name="S&P 500 Buy & Hold", line=dict(color=MC, width=1.5, dash="solid"),
        opacity=0.7,
    ))
    # shaded regimes
    for s, e, lbl, col in [
        ("2020-02-01", "2020-04-30", "COVID Crash", SC),
        ("2022-01-01", "2022-12-31", "Rate Hike 2022", MC),
    ]:
        fig.add_vrect(x0=s, x1=e, fillcolor=col, opacity=0.08, line_width=0,
                      annotation_text=lbl, annotation_position="top left",
                      annotation_font_color=col)
    fig.add_hline(y=1.0, line_dash="dot", line_color=GR, line_width=0.7)
    fig.update_layout(
        title="Backtest Equity Curves – $1.00 Initial NAV",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=450,
    )
    st.plotly_chart(fig, width="stretch")

    # Drawdown
    fig2 = go.Figure()
    for bt, col, name, nav_col in [
        (bt_b, BC, "Base",           "nav"),
        (bt_n, LC, "Regime-Neutral", "nav"),
        (bt_v, PC, "Vol Target",     "nav"),
        (bt_k, OC, "Kelly",          "nav"),
        (bt_b, MC, "S&P 500",        "nav_market"),
    ]:
        nav = bt[nav_col]
        dd  = (nav - nav.cummax()) / nav.cummax()
        # Convert #rrggbb → rgba(r,g,b,0.3) properly
        if col.startswith("#") and len(col) == 7:
            r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            fc = f"rgba({r},{g},{b},0.3)"
        else:
            fc = col
        fig2.add_trace(go.Scatter(
            x=bt["date"], y=dd, name=name, fill="tozeroy",
            line=dict(color=col, width=1.2),
            fillcolor=fc,
        ))
    fig2.update_layout(
        title="Drawdown — All Versions",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d", tickformat=".1%"),
        height=320,
    )
    st.plotly_chart(fig2, width="stretch")

# ---------- Tab 2: Monthly Returns ----------
with tab2:
    bt2 = bt_v[["date", "pnl_net"]].copy()
    bt2["year"]  = bt2["date"].dt.year
    bt2["month"] = bt2["date"].dt.month
    piv = bt2.pivot_table(index="year", columns="month", values="pnl_net", aggfunc="sum")
    MNAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    piv.columns = [MNAMES[int(c)-1] for c in piv.columns]

    fig3 = px.imshow(
        piv * 100,
        color_continuous_scale=["#f85149", "#0d1117", "#3fb950"],
        color_continuous_midpoint=0,
        text_auto=".1f",
        aspect="auto",
        title="Monthly Returns Heatmap – Vol-Targeted Strategy (%)",
    )
    fig3.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        coloraxis_colorbar=dict(tickformat=".1f", title="Ret %"),
        height=400,
    )
    st.plotly_chart(fig3, width="stretch")

# ---------- Tab 3: Factor Signals ----------
with tab3:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig4 = go.Figure()
        for fc, col, lbl in zip(
            FCOLS,
            [LC, BC, MC, SC],
            ["Value (inv-RSI+SMA)", "Momentum (12-1m)", "Quality (VPT)", "Low-Vol"],
        ):
            if fc in m.columns:
                fig4.add_trace(go.Scatter(
                    x=m["date"], y=m[fc], name=lbl,
                    line=dict(color=col, width=1.4), opacity=0.9,
                ))
        fig4.add_hline(y=0, line_dash="dot", line_color=GR)
        fig4.update_layout(
            title="Factor Z-Scores Over Time",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            height=360,
        )
        st.plotly_chart(fig4, width="stretch")

    with col_b:
        # Latest factor values bar chart
        latest = m.dropna(subset=FCOLS).iloc[-1]
        factor_labels = ["Value", "Momentum", "Quality", "Low-Vol"]
        factor_vals   = [float(latest[f]) for f in FCOLS]
        bar_colors    = [LC if v >= 0 else SC for v in factor_vals]
        figb = go.Figure(go.Bar(
            x=factor_labels, y=factor_vals,
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in factor_vals],
            textposition="outside",
        ))
        figb.update_layout(
            title="Latest Factor Z-Scores",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3",
            yaxis=dict(gridcolor="#21262d"),
            height=360,
        )
        st.plotly_chart(figb, width="stretch")

    # Signal comparison
    fig5 = go.Figure()
    sig_b = bt_b["sig_lag"].fillna(0).values
    fig5.add_trace(go.Bar(
        x=bt_b["date"], y=sig_b, name="Base signal",
        marker_color=[LC if v >= 0 else SC for v in sig_b],
        opacity=0.5,
    ))
    fig5.add_trace(go.Scatter(x=bt_v["date"], y=bt_v["sig_lag"].fillna(0),
                              name="Vol-Target", line=dict(color=PC, width=1.4)))
    fig5.add_trace(go.Scatter(x=bt_k["date"], y=bt_k["sig_lag"].fillna(0),
                              name="Kelly", line=dict(color=OC, width=1.4, dash="dash")))
    fig5.add_hline(y=0, line_color=GR, line_width=0.7)
    fig5.update_layout(
        title="Signal Comparison (Long > 0 | Short < 0)",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=300,
    )
    st.plotly_chart(fig5, width="stretch")

# ---------- Tab 4: Volatility ----------
with tab4:
    if "forecast_vol" in bt_v.columns and "vol_scale" in bt_v.columns:
        fig6 = make_subplots(specs=[[{"secondary_y": True}]])
        fig6.add_trace(go.Scatter(x=bt_v["date"], y=bt_v["real_vol_m"],
                                   name="Realised Vol", line=dict(color=SC, width=1.5)),
                       secondary_y=False)
        fig6.add_trace(go.Scatter(x=bt_v["date"], y=bt_v["forecast_vol"],
                                   name="Forecast Vol (EWMA)", line=dict(color=OC, width=1.5, dash="dash")),
                       secondary_y=False)
        fig6.add_hline(y=target_vol, line_dash="dot", line_color=LC,
                       annotation_text=f"Target {target_vol*100:.0f}%")
        fig6.add_trace(go.Scatter(x=bt_v["date"], y=bt_v["vol_scale"],
                                   name="Position Scale", line=dict(color=PC, width=1.3)),
                       secondary_y=True)
        fig6.update_yaxes(title_text="Vol (ann.)", tickformat=".0%", secondary_y=False)
        fig6.update_yaxes(title_text="Scale (x)", secondary_y=True)
        fig6.update_layout(
            title="Volatility Targeting — Forecast Vol & Scale",
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
            xaxis=dict(gridcolor="#21262d"), height=380,
        )
        st.plotly_chart(fig6, width="stretch")

    # Rolling 12m Sharpe
    fig7 = go.Figure()
    for bt, col, name, pnl_col in [
        (bt_b, BC, "Base",           "pnl_net"),
        (bt_n, LC, "Regime-Neutral", "pnl_net"),
        (bt_v, PC, "Vol Target",     "pnl_net"),
        (bt_k, OC, "Kelly",          "pnl_net"),
        (bt_b, MC, "S&P 500",        "pnl_market"),
    ]:
        rs = (bt[pnl_col].rolling(12).mean()
              / (bt[pnl_col].rolling(12).std() + 1e-9) * np.sqrt(12))
        fig7.add_trace(go.Scatter(x=bt["date"], y=rs, name=name,
                                   line=dict(color=col, width=1.5)))
    fig7.add_hline(y=0,   line_dash="dot", line_color=GR)
    fig7.add_hline(y=1.0, line_dash="dot", line_color=LC, opacity=0.4)
    fig7.update_layout(
        title="Rolling 12-Month Sharpe Ratio",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3", legend=dict(bgcolor="#161b22"),
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        height=350,
    )
    st.plotly_chart(fig7, width="stretch")

# ---------- Tab 5: Factor Correlations ----------
with tab5:
    fdata = m[FCOLS].dropna()
    corr  = fdata.corr().round(3)
    corr.index   = ["Value", "Momentum", "Quality", "LowVol"]
    corr.columns = ["Value", "Momentum", "Quality", "LowVol"]

    cv = corr.values.astype(float)
    mask = np.triu(np.ones_like(cv, dtype=bool), k=1)
    cv[mask] = np.nan

    fig8 = px.imshow(
        cv,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        color_continuous_scale=["#f85149", "#161b22", "#3fb950"],
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
        title="Factor Correlations (Lower Triangle)",
    )
    fig8.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        coloraxis_colorbar=dict(title="Corr"),
        height=380,
    )
    st.plotly_chart(fig8, width="stretch")

    st.markdown("**Factor Correlation Table**")
    st.dataframe(corr, width="stretch")

st.divider()

# ─────────────────────────────────────────
#  SECTION 4: Factor explanation
# ─────────────────────────────────────────
st.markdown("## 📖 Strategy Legend")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"""<div class='metric-card'>
        <div style='color:{LC}; font-size:18px; font-weight:700;'>📊 Value</div>
        <div style='color:#8b949e; font-size:12px; margin-top:6px;'>
        Inverse RSI + Price vs 12-month SMA. High score = cheap/oversold asset.
        </div></div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""<div class='metric-card'>
        <div style='color:{BC}; font-size:18px; font-weight:700;'>🚀 Momentum</div>
        <div style='color:#8b949e; font-size:12px; margin-top:6px;'>
        12-1 month return. High score = strong recent upward trend.
        </div></div>""", unsafe_allow_html=True)
with col_c:
    st.markdown(f"""<div class='metric-card'>
        <div style='color:{MC}; font-size:18px; font-weight:700;'>✅ Quality</div>
        <div style='color:#8b949e; font-size:12px; margin-top:6px;'>
        Volume-Price Trend (VPT). High score = volume confirms price moves.
        </div></div>""", unsafe_allow_html=True)
with col_d:
    st.markdown(f"""<div class='metric-card'>
        <div style='color:{SC}; font-size:18px; font-weight:700;'>🧘 Low-Vol</div>
        <div style='color:#8b949e; font-size:12px; margin-top:6px;'>
        Negative realised vol. High score = calm, stable price action.
        </div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#8b949e; font-size:12px; text-align:center;'>
⚠️ For educational purposes only. Not financial advice. Past performance is not indicative of future results.
</p>
""", unsafe_allow_html=True)
