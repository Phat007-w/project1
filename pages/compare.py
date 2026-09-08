import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go

# ==========================
# 🎯 Page Config
# ==========================
st.set_page_config(page_title="Compare Stocks • เปรียบเทียบหุ้น", page_icon="📈", layout="wide")
st.title("📈 เปรียบเทียบหุ้น 2 ตัว / Compare Two Stocks")

# ==========================
# 🧩 Stock Selection
# ==========================
preset = [
    "AAPL","MSFT","GOOGL","TSLA","NVDA",
    "PTT.BK","CPALL.BK","ADVANC.BK","KBANK.BK","SCC.BK",
    "^SET","^GSPC","^IXIC"
]

c1, c2 = st.columns(2)
with c1:
    ticker1 = st.selectbox("หุ้นตัวที่ 1 / Stock 1", preset, index=0)
with c2:
    ticker2 = st.selectbox("หุ้นตัวที่ 2 / Stock 2", preset, index=1)

if ticker1 == ticker2:
    st.warning("⚠️ โปรดเลือกหุ้นคนละตัว")
    st.stop()

# ==========================
# 📅 Date Range Selection
# ==========================
st.markdown("### 🗓 เลือกช่วงเวลา")

col_range, col_custom = st.columns([1, 2])

with col_range:
    range_option = st.radio(
        "เลือกช่วงเวลาแบบเร็ว",
        ["1M", "3M", "6M", "1Y", "3Y", "Max"],
        horizontal=True
    )

# คำนวณวันเริ่มต้นตามช่วงที่เลือก
today = date.today()
if range_option == "1M":
    start = today - timedelta(days=30)
elif range_option == "3M":
    start = today - timedelta(days=90)
elif range_option == "6M":
    start = today - timedelta(days=180)
elif range_option == "1Y":
    start = today - timedelta(days=365)
elif range_option == "3Y":
    start = today - timedelta(days=3*365)
else:  # Max
    start = today - timedelta(days=10*365)  # 10 ปีหลังสุด

with col_custom:
    start = st.date_input("เริ่มต้น / Start", start)
    end = st.date_input("สิ้นสุด / End", today)

if start >= end:
    st.error("❌ วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
    st.stop()

# ==========================
# 📊 Fetch Data
# ==========================
@st.cache_data
def fetch_data(ticker, start, end):
    """โหลดข้อมูลจาก Yahoo Finance"""
    t = yf.Ticker(ticker)
    hist = t.history(
        start=pd.to_datetime(start),
        end=pd.to_datetime(end) + pd.Timedelta(days=1),
        interval="1d"
    )
    hist = hist[["Close"]].rename(columns={"Close": ticker})
    return hist

try:
    df1 = fetch_data(ticker1, start, end)
    df2 = fetch_data(ticker2, start, end)
except Exception as e:
    st.error(f"ไม่สามารถโหลดข้อมูลหุ้นได้: {e}")
    st.stop()

# รวมข้อมูล
df = pd.concat([df1, df2], axis=1).dropna()

if df.empty:
    st.warning("⚠️ ไม่มีข้อมูลในช่วงนี้")
    st.stop()

# ==========================
# 📈 Metrics
# ==========================
st.markdown("### 📊 สรุปข้อมูล")

last_date = df.index[-1].strftime("%Y-%m-%d")
first_date = df.index[0].strftime("%Y-%m-%d")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{ticker1} (Last: {last_date})", f"{df[ticker1].iloc[-1]:,.2f}")
c2.metric(f"{ticker2} (Last: {last_date})", f"{df[ticker2].iloc[-1]:,.2f}")
c3.metric(f"%Δ {ticker1} ({first_date}→{last_date})", f"{((df[ticker1].iloc[-1]-df[ticker1].iloc[0])/df[ticker1].iloc[0]*100):+.2f}%")
c4.metric(f"%Δ {ticker2} ({first_date}→{last_date})", f"{((df[ticker2].iloc[-1]-df[ticker2].iloc[0])/df[ticker2].iloc[0]*100):+.2f}%")

st.markdown("---")

# ==========================
# 📉 Plot Comparison
# ==========================
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[ticker1], mode='lines', name=ticker1))
fig.add_trace(go.Scatter(x=df.index, y=df[ticker2], mode='lines', name=ticker2))
fig.update_layout(
    title=f"📊 Comparison: {ticker1} vs {ticker2}",
    xaxis_title="Date",
    yaxis_title="Price",
    height=550,
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# 🔍 Correlation
# ==========================
corr = df[ticker1].pct_change().corr(df[ticker2].pct_change())
st.info(f"📈 **Correlation (daily returns)** between `{ticker1}` & `{ticker2}`: **{corr:.4f}**")

# ==========================
# 💾 Download CSV
# ==========================
csv = df.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    f"⬇️ ดาวน์โหลด CSV {ticker1} + {ticker2}",
    data=csv,
    file_name=f"{ticker1}_{ticker2}_comparison.csv",
    mime="text/csv"
)
