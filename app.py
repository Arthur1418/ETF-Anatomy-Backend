from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)
CORS(app)

ETF_LIST = ["SPY", "QQQ", "VTI", "IWM"]
MODELS = {}
DATA = {}
FEATURES = ['Open', 'High', 'Low', 'Volume', 'SMA_50', 'SMA_200', 'Volatility']
TARGET = 'Adj Close'

# --- Data and Model Preparation ---
def load_data(symbol):
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=5*365)
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    df['Return'] = df['Adj Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=21).std() * np.sqrt(252)
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    return df

# Train models at startup
for symbol in ETF_LIST:
    df = load_data(symbol)
    DATA[symbol] = df
    X = df[FEATURES]
    y = df[TARGET]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    MODELS[symbol] = model

# --- Helper Functions ---
def get_latest_features(df):
    return df[FEATURES].iloc[-1].values.reshape(1, -1)

# RSI calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- API Endpoints ---

@app.route('/api/predict/<symbol>')
def predict(symbol):
    symbol = symbol.upper()
    if symbol not in ETF_LIST:
        return jsonify({'error': 'Symbol not supported'}), 400
    df = DATA[symbol]
    model = MODELS[symbol]
    features = get_latest_features(df)
    pred = model.predict(features)[0]
    return jsonify({'symbol': symbol, 'predicted_price': round(float(pred), 2)})

@app.route('/api/risk/<symbol>')
def risk(symbol):
    symbol = symbol.upper()
    if symbol not in ETF_LIST:
        return jsonify({'error': 'Symbol not supported'}), 400
    df = DATA[symbol]
    latest_vol = round(df['Volatility'].iloc[-1], 4)
    sharpe = round(df['Return'].mean() / df['Return'].std() * np.sqrt(252), 2)
    return jsonify({'symbol': symbol, 'volatility': latest_vol, 'sharpe_ratio': sharpe})

@app.route('/api/explain/<symbol>')
def explain(symbol):
    symbol = symbol.upper()
    if symbol not in ETF_LIST:
        return jsonify({'error': 'Symbol not supported'}), 400
    df = DATA[symbol]
    rsi_series = compute_rsi(df['Adj Close'])
    latest_rsi = round(rsi_series.iloc[-1], 1)
    ma50 = df['SMA_50'].iloc[-1]
    ma200 = df['SMA_200'].iloc[-1]
    price = df['Adj Close'].iloc[-1]
    crossover = 'above' if price > ma50 else 'below'
    volume_spike = df['Volume'].iloc[-1] > df['Volume'].rolling(window=20).mean().iloc[-1] * 1.5
    explanation = []
    explanation.append(f"RSI (14) is {latest_rsi}.")
    explanation.append(f"Price is {crossover} 50-day MA.")
    if volume_spike:
        explanation.append("Recent volume spike detected.")
    return jsonify({'symbol': symbol, 'explanation': explanation})

@app.route('/etf/<symbol>')
def etf_dashboard(symbol):
    symbol = symbol.upper()
    if symbol not in ETF_LIST:
        return jsonify({'error': 'Symbol not supported'}), 400

    df = DATA[symbol]
    model = MODELS[symbol]
    latest_price = df['Adj Close'].iloc[-1]
    prev_price = df['Adj Close'].iloc[-2]
    change = round((latest_price - prev_price) / prev_price * 100, 2)

    signal = "buy" if change > 0 else "sell"
    confidence = int(abs(change) * 10)

    chart_dates = df.index[-30:].strftime('%Y-%m-%d').tolist()
    chart_prices = df['Adj Close'].iloc[-30:].round(2).tolist()

    return jsonify({
        "price": round(float(latest_price), 2),
        "changePercent": change,
        "signal": signal,
        "confidence": confidence,
        "chart": {
            "dates": chart_dates,
            "prices": chart_prices
        }
    })

# --- Start App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
