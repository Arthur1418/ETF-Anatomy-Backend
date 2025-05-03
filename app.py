import os
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
app.config["DEBUG"] = True  # Enable debug mode

# List of ETF symbols for prediction
symbols = ['SPY', 'QQQ', 'VTI', 'IWM']

# Function to load the ETF data
def load_data(symbol):
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=5*365)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    
    # Print out the first few rows of the data for debugging
    print(f"Data for {symbol}:")
    print(df.head())  # Print the first few rows of data to inspect its structure

    # Flatten the MultiIndex if present and print available columns
    df.columns = df.columns.droplevel(1)  # Remove multi-level column names
    print(f"Loaded data columns for {symbol}: {df.columns}")

    # Ensure we are using 'Close' instead of 'Adj Close'
    if 'Adj Close' in df.columns:
        df.drop(columns=['Adj Close'], inplace=True)

    # Use 'Close' for calculations
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=21).std() * np.sqrt(252)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df.dropna(inplace=True)
    return df

# Function to train a model for each symbol
def train_model(symbol):
    df = load_data(symbol)
    
    # Prepare data for training
    X = df[['Volatility', 'SMA_50', 'SMA_200']]
    y = (df['Return'] > 0).astype(int)  # 1 if the return is positive, 0 if negative

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Save the model
    model_filename = f"{symbol}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Model for {symbol} trained and stored.")

# Train models for each ETF symbol
for symbol in symbols:
    train_model(symbol)

# Function to make a prediction for a given symbol
def make_prediction(symbol):
    df = load_data(symbol)
    X = df[['Volatility', 'SMA_50', 'SMA_200']].iloc[-1:].values
    model_filename = f"{symbol}_model.pkl"
    
    # Load the trained model
    model = joblib.load(model_filename)
    
    # Make a prediction (0 = sell, 1 = buy)
    prediction = model.predict(X)
    
    return 'Buy' if prediction == 1 else 'Sell'

@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the ETF prediction API!'})

@app.route('/predict/<symbol>')
def predict(symbol):
    if symbol not in symbols:
        return jsonify({'error': 'Invalid symbol. Available symbols are: SPY, QQQ, VTI, IWM.'})

    prediction = make_prediction(symbol)
    return jsonify({'symbol': symbol, 'prediction': prediction})

if __name__ == '__main__':
    app.run()
