import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from streamlit_autorefresh import st_autorefresh

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Quantitative AI Trading Framework (Academic Version)",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# 2. SIDEBAR CONFIGURATION
# =====================================================
with st.sidebar:
    st.header("⚙️ System Parameters")

    st.subheader("🔮 Forecasting Settings")
    forecast_days = st.slider("Forecast Horizon (Days)", min_value=3, max_value=60, value=14, step=1)

    st.subheader("💼 Backtest Assumptions")
    commission_pct = st.number_input("Commission Fee (%)", value=0.1, step=0.01) / 100.0
    slippage_pct = st.number_input("Slippage (%)", value=0.05, step=0.01) / 100.0

    st.subheader("🔄 Refresh Rate")
    refresh_rate = st.selectbox("Auto Refresh", ["Off", "Every 30s", "Every 1m"], index=0)

refresh_map = {"Off": None, "Every 30s": 30_000, "Every 1m": 60_000}

if "page" not in st.session_state:
    st.session_state.page = "table"
if "selected" not in st.session_state:
    st.session_state.selected = None

if st.session_state.page == "table" and refresh_map[refresh_rate]:
    st_autorefresh(interval=refresh_map[refresh_rate], key="scanner_refresh")

# =====================================================
# 3. FEATURE ENGINEERING & INDICATORS
# =====================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def add_technical_features(df, ema_fast=9, ema_slow=21):
    df = df.copy()
    df["EMA_FAST"] = df["Close"].ewm(span=ema_fast).mean()
    df["EMA_SLOW"] = df["Close"].ewm(span=ema_slow).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
    df["Return_5"] = df["Close"].pct_change(5) * 100
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()

    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_UPPER"] = sma20 + (2 * std20)
    df["BB_LOWER"] = sma20 - (2 * std20)
    df["BB_PCT"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"] + 1e-9)
    df["VOLATILITY"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    return df

# =====================================================
# 4. ADVANCED FORECASTING & CROSS-VALIDATION
# =====================================================
def predict_future_price_ridge(df, days):
    try:
        recent_df = df.tail(60).copy()
        recent_df['ID'] = range(len(recent_df))
        X, y = recent_df[['ID']], recent_df['Close']

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
        future_X = np.array(range(len(recent_df), len(recent_df) + days)).reshape(-1, 1)
        future_prices = model.predict(future_X)

        residuals = y - model.predict(X)
        std_err = np.std(residuals)

        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecast': future_prices,
            'Upper_Bound': future_prices + (1.96 * std_err),
            'Lower_Bound': future_prices - (1.96 * std_err)
        }).set_index('Date')
        return forecast_df
    except Exception:
        return pd.DataFrame()

def train_and_evaluate_ml_cv(df, n_splits=5):
    """5-Fold TimeSeriesCrossValidation Evaluation"""
    df = df.copy()
    df["EMA_DIFF"] = df["EMA_FAST"] - df["EMA_SLOW"]
    df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
    df["VOL_RATIO"] = df["Volume"] / (df["VOL_AVG20"] + 1e-9)
    df["RSI_NORM"] = df["RSI"] / 100.0
    df["FUTURE"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.02).astype(int)

    features = ["EMA_DIFF", "RSI_NORM", "MACD_DIFF", "Return_5", "VOL_RATIO", "BB_PCT", "VOLATILITY"]
    train_data = df.dropna(subset=features + ["FUTURE"]).copy()

    if len(train_data) < 80:
        return 0.0, {}, None, features

    X = train_data[features]
    y = train_data["FUTURE"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, precs, recs, f1s = [], [], [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)

        accs.append(accuracy_score(y_va, preds))
        precs.append(precision_score(y_va, preds, zero_division=0))
        recs.append(recall_score(y_va, preds, zero_division=0))
        f1s.append(f1_score(y_va, preds, zero_division=0))

    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X, y)

    metrics = {
        "Mean Accuracy": round(np.mean(accs) * 100, 2),
        "Mean Precision": round(np.mean(precs) * 100, 2),
        "Mean Recall": round(np.mean(recs) * 100, 2),
        "Mean F1-Score": round(np.mean(f1s) * 100, 2)
    }

    current_X = df.iloc[[-1]][features].fillna(0)
    current_prob = round(final_model.predict_proba(current_X)[0][1] * 100, 2) if len(final_model.classes_) > 1 else 0.0

    return current_prob, metrics, final_model, features

# =====================================================
# 5. BACKTEST & SENSITIVITY ANALYSIS
# =====================================================
def run_academic_backtest(df, capital, risk_p, rr_ratio, commission, slippage):
    balance = capital
    equity = [capital]
    trades = []
    in_position = False
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    start_idx = 30

    for i in range(start_idx, len(df)):
        curr = df.iloc[i]
        date_curr = df.index[i]

        buy_signal = (curr["EMA_FAST"] > curr["EMA_SLOW"] and curr["MACD"] > curr["MACD_SIGNAL"] and
                      45 < curr["RSI"] < 70 and not in_position)

        if buy_signal:
            candidate_stop_loss = curr["EMA_SLOW"] * 0.98
            risk_amt = balance * (risk_p / 100)
            risk_per_share = curr["Close"] - candidate_stop_loss
            if risk_per_share > 0:
                candidate_shares = int(risk_amt / risk_per_share)
                if candidate_shares > 0:
                    candidate_entry_price = curr["Close"] * (1 + slippage)
                    cost = candidate_shares * candidate_entry_price
                    fee = cost * commission
                    if balance >= (cost + fee):
                        balance -= (cost + fee)
                        in_position = True
                        shares = candidate_shares
                        entry_price = candidate_entry_price
                        stop_loss = candidate_stop_loss
                        trades.append({"Type": "BUY", "Date": date_curr, "Price": entry_price, "Fee": fee})

        elif in_position:
            exit_price = 0
            reason = ""
            if curr["Low"] <= stop_loss:
                exit_price = stop_loss * (1 - slippage)
                reason = "Stop Loss"
            elif curr["EMA_FAST"] < curr["EMA_SLOW"]:
                exit_price = curr["Close"] * (1 - slippage)
                reason = "Trend Change"

            if reason:
                revenue = shares * exit_price
                fee = revenue * commission
                balance += (revenue - fee)
                profit = (exit_price - entry_price) * shares - fee
                trades.append({"Type": "SELL", "Date": date_curr, "Price": exit_price, "Reason": reason, "Profit": profit, "Fee": fee})
                in_position = False
                shares = 0
                entry_price = 0.0

        equity.append(balance + (shares * curr["Close"] if in_position else 0))

    trades_df = pd.DataFrame(trades)
    eq_series = pd.Series(equity)
    returns = eq_series.pct_change().dropna()

    rf = 0.02 / 252
    excess_returns = returns - rf
    sharpe = np.sqrt(252) * (excess_returns.mean() / (returns.std() + 1e-9))
    cum_max = eq_series.cummax()
    max_drawdown = ((eq_series - cum_max) / cum_max).min() * 100

    return trades_df, equity, round(sharpe, 2), round(max_drawdown, 2)

def generate_sensitivity_analysis(df, capital, risk_p, rr_ratio, slippage):
    """Analyze impact of varying Commission rates on Sharpe Ratio"""
    commissions = np.linspace(0.000, 0.005, 10)  # 0.0% to 0.5%
    sharpes = []

    for comm in commissions:
        _, _, sharpe, _ = run_academic_backtest(df, capital, risk_p, rr_ratio, comm, slippage)
        sharpes.append(sharpe)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=commissions * 100, y=sharpes, mode='lines+markers', line=dict(color='#00CC96', width=2)))
    fig.update_layout(
        title="Sensitivity Analysis: Commission Rate vs. Sharpe Ratio",
        xaxis_title="Commission Rate (%)",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        height=350
    )
    return fig

def estimate_dividend_frequency_months(div_history):
    """ประมาณความถี่การจ่ายปันผลจากประวัติย้อนหลัง (หน่วย: ทุกกี่เดือน)"""
    if div_history is None or len(div_history) < 2:
        return None

    # ใช้ 8 ครั้งล่าสุดพอ ป้องกัน outlier จากประวัติเก่าที่นโยบายอาจเปลี่ยนไปแล้ว
    recent = div_history.tail(8)
    diffs_days = recent.index.to_series().diff().dropna().dt.days
    if diffs_days.empty:
        return None

    avg_days = diffs_days.mean()
    avg_months = avg_days / 30.44
    return round(avg_months, 1)

@st.cache_data(ttl=3600)
def get_dividend_data(ticker, current_price=None):
    """
    ใช้มาตรฐาน 'Indicated Annual Dividend' (แบบเดียวกับ Yahoo/Robinhood):
    เงินปันผลงวดล่าสุด x จำนวนครั้งที่จ่ายต่อปี
    (ไม่ใช้วิธีรวมยอดจ่ายจริงย้อนหลัง 365 วัน เพราะถ้าจ่ายทุก ~91 วัน
    บางทีจะมี 5 งวดหลุดเข้ามาในหน้าต่าง 365 วันพอดี ทำให้ยอดพองเกินจริง)
    """
    stock = yf.Ticker(ticker)
    div_history = stock.dividends

    ttm_dividend = 0.0
    freq_months = None

    if div_history is not None and len(div_history) > 0:
        freq_months = estimate_dividend_frequency_months(div_history)
        latest_div = float(div_history.iloc[-1])

        if freq_months and freq_months > 0:
            payments_per_year = max(round(12 / freq_months), 1)
        else:
            payments_per_year = 1  # ข้อมูลไม่พอจะประมาณความถี่ ถือว่าจ่ายปีละครั้ง

        ttm_dividend = round(latest_div * payments_per_year, 4)

    if current_price and current_price > 0 and ttm_dividend > 0:
        div_yield = (ttm_dividend / current_price) * 100
    else:
        div_yield = 0.0

    return {
        "rate": round(ttm_dividend, 2),
        "yield": round(div_yield, 2),
        "freq_months": freq_months,
        "history": div_history
    }

def run_academic_backtest_with_dividends(df, capital, risk_p, rr_ratio, commission, slippage, div_history):
    balance = capital
    equity = [capital]
    total_dividends_received = 0.0
    in_position = False
    shares = 0
    stop_loss = 0.0
    start_idx = 30

    for i in range(start_idx, len(df)):
        date_curr = df.index[i]
        curr = df.iloc[i]

        # 1. เช็กการได้รับเงินปันผล (ถ้าถือหุ้นอยู่ ณ วัน XD)
        if in_position and div_history is not None and date_curr in div_history.index:
            div_per_share = div_history.loc[date_curr]
            payout = shares * div_per_share
            balance += payout
            total_dividends_received += payout

        # 2. เงื่อนไขการเข้าซื้อ (BUY)
        buy_signal = (curr["EMA_FAST"] > curr["EMA_SLOW"] and curr["MACD"] > curr["MACD_SIGNAL"] and
                      45 < curr["RSI"] < 70 and not in_position)

        if buy_signal:
            candidate_stop_loss = curr["EMA_SLOW"] * 0.98
            risk_amt = balance * (risk_p / 100)
            risk_per_share = curr["Close"] - candidate_stop_loss
            if risk_per_share > 0:
                candidate_shares = int(risk_amt / risk_per_share)
                if candidate_shares > 0:
                    entry_price = curr["Close"] * (1 + slippage)
                    cost = candidate_shares * entry_price
                    fee = cost * commission
                    if balance >= (cost + fee):
                        balance -= (cost + fee)
                        in_position = True
                        shares = candidate_shares
                        stop_loss = candidate_stop_loss

        # 3. เงื่อนไขการขาย (SELL)
        elif in_position:
            exit_price = 0
            reason = ""
            if curr["Low"] <= stop_loss:
                exit_price = stop_loss * (1 - slippage)
                reason = "Stop Loss"
            elif curr["EMA_FAST"] < curr["EMA_SLOW"]:
                exit_price = curr["Close"] * (1 - slippage)
                reason = "Trend Change"

            if reason:
                revenue = shares * exit_price
                fee = revenue * commission
                balance += (revenue - fee)
                in_position = False
                shares = 0

        equity.append(balance + (shares * curr["Close"] if in_position else 0))

    return equity, round(total_dividends_received, 2)

@st.cache_data(ttl=300)
def get_price_snapshot(ticker):
    """ดึงราคาเปิด, ราคาปิดล่าสุด, และราคานอกเวลาทำการ (pre-market หรือ after-hours ตามช่วงเวลาจริง)"""
    stock = yf.Ticker(ticker)
    info = stock.info

    open_price = info.get("regularMarketOpen") or info.get("open")
    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
    last_close = info.get("regularMarketPrice") or info.get("currentPrice") or prev_close

    # marketState บอกช่วงเวลาปัจจุบันจริงๆ: PRE, REGULAR, POST, POSTPOST, CLOSED
    market_state = info.get("marketState", "")

    pre_market_price = info.get("preMarketPrice")
    pre_market_change = info.get("preMarketChange")
    pre_market_change_pct = info.get("preMarketChangePercent")

    post_market_price = info.get("postMarketPrice")
    post_market_change = info.get("postMarketChange")
    post_market_change_pct = info.get("postMarketChangePercent")

    # เลือกให้ตรงกับช่วงเวลาจริง: ถ้าเป็น pre-market ให้ใช้ pre, นอกนั้นถ้ามี post ให้ใช้ post
    if market_state == "PRE" and pre_market_price is not None:
        extended_label = "ราคาพรีมาร์เก็ต (Pre-Market)"
        extended_price = pre_market_price
        extended_change = pre_market_change
        extended_change_pct = pre_market_change_pct
    elif post_market_price is not None:
        extended_label = "ราคาหลังตลาดปิด (After-Hours)"
        extended_price = post_market_price
        extended_change = post_market_change
        extended_change_pct = post_market_change_pct
    elif pre_market_price is not None:
        extended_label = "ราคาพรีมาร์เก็ต (Pre-Market)"
        extended_price = pre_market_price
        extended_change = pre_market_change
        extended_change_pct = pre_market_change_pct
    else:
        extended_label = "ราคานอกเวลาทำการ"
        extended_price = None
        extended_change = None
        extended_change_pct = None

    return {
        "open": open_price,
        "last_close": last_close,
        "market_state": market_state,
        "extended_label": extended_label,
        "extended_price": extended_price,
        "extended_change": extended_change,
        "extended_change_pct": extended_change_pct,
    }

# =====================================================
# 6. APP CONTROLLER & MAIN UI
# =====================================================
@st.cache_data(ttl=3600)
def fetch_and_prepare_data(ticker, start, end, ema_fast, ema_slow):
    df = yf.download(ticker, start=start - timedelta(days=100), end=end + timedelta(days=1), progress=False)
    if df.empty or len(df) < 60:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return add_technical_features(df, ema_fast, ema_slow)

start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())
capital = st.sidebar.number_input("Capital ($)", 10000)
risk_p = st.sidebar.slider("Risk per trade (%)", 0.5, 5.0, 2.0)
style = st.sidebar.radio("Strategy Style", ["Day Trade", "Swing Trade"])

ema_fast, ema_slow, rr = (9, 21, 1.5) if style == "Day Trade" else (20, 50, 2.0)
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "AVGO"]

if st.session_state.page == "table":
    st.title("📈 Quantitative AI Market Scanner")
    st.caption("Robust Multi-Factor Technical Screening & Machine Learning Pipeline")

    results = []
    with st.spinner("Scanning universe & executing quantitative analysis..."):
        for s in STOCKS:
            df = fetch_and_prepare_data(s, start_date, end_date, ema_fast, ema_slow)
            if df is not None:
                last = df.iloc[-1]
                score = 0
                if last["EMA_FAST"] > last["EMA_SLOW"]:
                    score += 2
                if last["Return_5"] > 0:
                    score += 1
                if last["Volume"] > last["VOL_AVG20"]:
                    score += 1
                if last["MACD"] > last["MACD_SIGNAL"]:
                    score += 1

                results.append({"Ticker": s, "Price": round(float(last["Close"]), 2), "Score": score, "RSI": round(float(last["RSI"]), 2)})

    if results:
        df_res = pd.DataFrame(results).sort_values("Score", ascending=False)
        st.dataframe(df_res, use_container_width=True)
        cols = st.columns(5)
        for i, t in enumerate(df_res["Ticker"]):
            if cols[i % 5].button(f"🔍 {t}", use_container_width=True):
                st.session_state.selected = t
                st.session_state.page = "chart"
                st.rerun()

elif st.session_state.page == "chart":
    t = st.session_state.selected
    df = fetch_and_prepare_data(t, start_date, end_date, ema_fast, ema_slow)
    forecast_df = predict_future_price_ridge(df, days=forecast_days)

    st.button("⬅ Back to Scanner", on_click=lambda: st.session_state.update(page="table"))

    st.title(f"📊 Academic Research Dashboard: {t}")

    # ---- ราคาเปิด / ราคาปิด / ราคานอกเวลาทำการ (pre-market หรือ after-hours ตามช่วงเวลาจริง) ----
    snap = get_price_snapshot(t)
    p1, p2, p3 = st.columns(3)
    p1.metric("ราคาเปิด", f"${snap['open']:.2f}" if snap['open'] is not None else "N/A")
    p2.metric("ราคาปิด", f"${snap['last_close']:.2f}" if snap['last_close'] is not None else "N/A")

    if snap["extended_price"] is not None:
        delta_str = None
        if snap["extended_change"] is not None and snap["extended_change_pct"] is not None:
            delta_str = f"{snap['extended_change']:+.2f} ({snap['extended_change_pct']:+.2f}%)"
        p3.metric(snap["extended_label"], f"${snap['extended_price']:.2f}", delta=delta_str)
    else:
        p3.metric(snap["extended_label"], "N/A")

    # ดึงข้อมูลปันผล (ย้ายมาไว้ตรงนี้ หลังจากมี t และ df แล้ว)
    # ส่งราคาปัจจุบันเข้าไปเพื่อคำนวณ yield เองจากปันผลจริง/ราคาจริง
    current_price = float(df['Close'].iloc[-1])
    div_info = get_dividend_data(t, current_price=current_price)

    st.subheader("💵 เงินปันผล")
    if div_info['rate'] == 0.0:
        st.info(f"ℹ️ {t} ไม่มีประวัติการจ่ายเงินปันผล")
    else:
        freq_text = f"ทุก ๆ {div_info['freq_months']} เดือน" if div_info['freq_months'] else "N/A"
        st.markdown(
            f"""
            <div style="
                border: 1px solid #7C3AED;
                border-radius: 16px;
                padding: 20px 24px;
                background-color: rgba(124, 58, 237, 0.08);
                margin-bottom: 8px;
            ">
                <div style="display:flex; justify-content:space-between; gap:24px;">
                    <div>
                        <div style="color:#B197FC; font-size:14px; margin-bottom:6px;">🪙 อัตราเงินปันผล</div>
                        <div style="color:#FFFFFF; font-size:26px; font-weight:700;">{div_info['yield']}%</div>
                        <div style="color:#8B8B9E; font-size:12px; margin-top:2px;">ของมูลค่าหุ้นที่ซื้อ (ต่อปี)</div>
                    </div>
                    <div>
                        <div style="color:#B197FC; font-size:14px; margin-bottom:6px;">💸 เงินปันผลต่อปี</div>
                        <div style="color:#FFFFFF; font-size:26px; font-weight:700;">{div_info['rate']:.2f} USD</div>
                        <div style="color:#8B8B9E; font-size:12px; margin-top:2px;">เงินปันผลย้อนหลัง 12 เดือน (ต่อหุ้น)</div>
                    </div>
                </div>
                <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.15); margin: 16px 0;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="color:#B197FC; font-size:14px;">ความถี่ในการจ่ายเงินปันผล</span>
                    <span style="color:#FFFFFF; font-size:15px; font-weight:600;">{freq_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    prob, cv_metrics, model, feats = train_and_evaluate_ml_cv(df, n_splits=5)

    tab1, tab2, tab3 = st.tabs([
        "🧠 5-Fold Cross-Validation Analysis",
        "🧪 Backtest & Sensitivity Study",
        "💰 Dividend-Adjusted Backtest"
    ])

    with tab1:
        st.subheader("📊 5-Fold TimeSeriesCrossValidation Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CV Mean Accuracy", f"{cv_metrics.get('Mean Accuracy', 0)}%")
        m2.metric("CV Mean Precision", f"{cv_metrics.get('Mean Precision', 0)}%")
        m3.metric("CV Mean Recall", f"{cv_metrics.get('Mean Recall', 0)}%")
        m4.metric("CV Mean F1-Score", f"{cv_metrics.get('Mean F1-Score', 0)}%")

        # Plot forecast graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'))
        if not forecast_df.empty:
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], line=dict(color='yellow', dash='dash'), name='Ridge Forecast'))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper_Bound'], line=dict(color='rgba(255,255,255,0.2)'), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower_Bound'], line=dict(color='rgba(255,255,255,0.2)'), fill='tonexty', showlegend=False))
        fig.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧪 Friction-Adjusted Backtest & Sensitivity Curve")
        trades, equity, sharpe, mdd = run_academic_backtest(df, capital, risk_p, rr, commission_pct, slippage_pct)

        c1, c2 = st.columns(2)
        c1.metric("Strategy Sharpe Ratio", f"{sharpe}")
        c2.metric("Max Drawdown (MDD)", f"{mdd}%")

        st.plotly_chart(generate_sensitivity_analysis(df, capital, risk_p, rr, slippage_pct), use_container_width=True)

    with tab3:
        st.subheader("💰 Backtest Including Dividend Reinvestment")
        div_equity, total_div = run_academic_backtest_with_dividends(
            df, capital, risk_p, rr, commission_pct, slippage_pct, div_info["history"]
        )

        e1, e2 = st.columns(2)
        e1.metric("Total Dividends Received", f"${total_div:,.2f}")
        e2.metric("Ending Equity (w/ Dividends)", f"${div_equity[-1]:,.2f}")

        fig_div = go.Figure()
        fig_div.add_trace(go.Scatter(y=div_equity, mode='lines', name='Equity (with Dividends)', line=dict(color='#FFA15A')))
        fig_div.update_layout(template="plotly_dark", height=350, xaxis_title="Bar Index", yaxis_title="Equity ($)")
        st.plotly_chart(fig_div, use_container_width=True)