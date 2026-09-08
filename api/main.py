from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from typing import List, Optional

app = FastAPI(title="Stock AI API")

# Setup CORS for React communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน Production ควรระบุเป็น URL ของ React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions (Ported from Home.py) ---

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def predict_future_price(df, days):
    try:
        recent_df = df.tail(30).copy()
        recent_df['ID'] = range(len(recent_df))
        X = recent_df[['ID']]
        y = recent_df['Close']
        model = LinearRegression()
        model.fit(X, y)
        last_date = df.index[-1]
        future_X = np.array(range(len(recent_df), len(recent_df) + days)).reshape(-1, 1)
        future_prices = model.predict(future_X)
        
        forecast = []
        for i, price in enumerate(future_prices):
            date = (last_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
            forecast.append({"date": date, "price": round(float(price), 2)})
        return forecast
    except:
        return []

def train_and_predict_ml(df):
    try:
        df = df.copy()
        df["EMA_FAST"] = df["Close"].ewm(span=9).mean()
        df["EMA_SLOW"] = df["Close"].ewm(span=21).mean()
        df["RSI"] = compute_rsi(df["Close"])
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()
        df["EMA_DIFF"] = df["EMA_FAST"] - df["EMA_SLOW"]
        df["MACD_DIFF"] = df["MACD"] - df["MACD_SIGNAL"]
        df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
        df["VOL_RATIO"] = df["Volume"] / (df["VOL_AVG20"] + 1e-5)
        df["RSI_NORM"] = df["RSI"] / 100.0
        df["Return_5"] = df["Close"].pct_change(5) * 100
        df["FUTURE"] = (df["Close"].shift(-5) / df["Close"] - 1 > 0.02).astype(int)
        
        features = ["EMA_DIFF", "RSI_NORM", "MACD_DIFF", "Return_5", "VOL_RATIO"]
        train_data = df.dropna(subset=features + ["FUTURE"]).copy()
        
        if len(train_data) < 50: return 0.0
        
        X = train_data[features]
        y = train_data["FUTURE"]
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        current_X = df.iloc[[-1]][features].fillna(0)
        prob = model.predict_proba(current_X)[0][1] * 100
        return round(float(prob), 2)
    except:
        return 0.0

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Stock AI API is running"}

@app.get("/api/stock/{ticker}")
async def get_stock_analysis(ticker: str, days: int = 14):
    try:
        # ดึงข้อมูลย้อนหลัง 1 ปี
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock not found")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # คำนวณ Indicators พื้นฐาน
        df["EMA_FAST"] = df["Close"].ewm(span=9).mean()
        df["EMA_SLOW"] = df["Close"].ewm(span=21).mean()
        df["RSI"] = compute_rsi(df["Close"])
        
        last_price = round(float(df["Close"].iloc[-1]), 2)
        prev_price = round(float(df["Close"].iloc[-2]), 2)
        change_pct = round(((last_price - prev_price) / prev_price) * 100, 2)
        
        # พยากรณ์ AI
        forecast = predict_future_price(df, days)
        ml_prob = train_and_predict_ml(df)
        
        # ตัดข้อมูลเพื่อส่งไปทำกราฟ (เช่น 60 วันล่าสุด)
        history = []
        chart_df = df.tail(60)
        for index, row in chart_df.iterrows():
            history.append({
                "date": index.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })
            
        return {
            "ticker": ticker,
            "current_price": last_price,
            "change_percent": change_pct,
            "ml_confidence": ml_prob,
            "forecast": forecast,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
