from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Sample ETF data (mocked)
ETF_DATA = {
    "niftybees": {
        "price": 215.85,
        "changePercent": 0.58,
        "signal": "BUY",
        "confidence": 76,
        "chart": {
            "dates": ["2024-04-20", "2024-04-21", "2024-04-22"],
            "prices": [210.3, 213.6, 215.85]
        }
    },
    "bankbees": {
        "price": 430.15,
        "changePercent": -0.42,
        "signal": "SELL",
        "confidence": 64,
        "chart": {
            "dates": ["2024-04-20", "2024-04-21", "2024-04-22"],
            "prices": [435.0, 432.0, 430.15]
        }
    },
    "itbees": {
        "price": 320.50,
        "changePercent": 0.78,
        "signal": "BUY",
        "confidence": 81,
        "chart": {
            "dates": ["2024-04-20", "2024-04-21", "2024-04-22"],
            "prices": [312.0, 318.9, 320.5]
        }
    },
    "psubankbees": {
        "price": 55.90,
        "changePercent": -0.12,
        "signal": "HOLD",
        "confidence": 50,
        "chart": {
            "dates": ["2024-04-20", "2024-04-21", "2024-04-22"],
            "prices": [56.3, 56.1, 55.9]
        }
    }
}

@app.route("/etf/<string:ticker>")
def get_etf_data(ticker):
    data = ETF_DATA.get(ticker.lower())
    if not data:
        return jsonify({"error": "ETF not found"}), 404
    return jsonify(data)

@app.route("/market")
def get_market_data():
    return jsonify({
        "sp500": 5105.75,
        "nasdaq": 16120.40,
        "vix": 13.45,
        "lastUpdated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/prediction")
def get_prediction_data():
    return jsonify({
        "labels": ["Jan", "Feb", "Mar", "Apr"],
        "data": [120, 135, 128, 145]
    })

@app.route("/api/etf/india")
def get_indian_etfs():
    return jsonify({k: v for k, v in ETF_DATA.items() if k in ["niftybees", "bankbees", "itbees", "psubankbees"]})

# Optional: If needed in future
@app.route("/api/etf/us")
def get_us_etfs():
    return jsonify({})  # Add US ETFs here

@app.route("/api/etf/global")
def get_global_etfs():
    return jsonify({})  # Add global ETFs here

if __name__ == "__main__":
    app.run(debug=True)
