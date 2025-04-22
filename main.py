from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = FastAPI()

# Define the model for the input data
class ETFRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str

# Load or initialize your model (you can replace this with a trained model)
def train_model(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()

    # Prepare the data for training the model
    data.dropna(inplace=True)
    X = np.array(range(len(data))).reshape(-1, 1)  # Time as the feature (simple linear regression)
    y = data['Returns'].values

    model = LinearRegression()
    model.fit(X, y)

    # Save the model for future use
    joblib.dump(model, f'{ticker}_model.pkl')

# Predict the ETF movement using the trained model
def predict_movement(ticker: str, start_date: str, end_date: str):
    # Load the trained model
    model = joblib.load(f'{ticker}_model.pkl')

    # Get the historical data for prediction
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    data.dropna(inplace=True)

    # Use the model to predict future returns
    X = np.array(range(len(data), len(data) + 1)).reshape(-1, 1)
    predicted_return = model.predict(X)

    return predicted_return[0]

# Generate a simple plot of the ETF performance over time
def generate_plot(ticker: str, start_date: str, end_date: str):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Adj Close'].plot(title=f'{ticker} Price Over Time')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.tight_layout()
    
    # Convert plot to PNG
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode PNG to base64
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return plot_base64

@app.get("/")
def read_root():
    return {"message": "Welcome to the ETF Prediction API!"}

@app.post("/predict")
def predict_etf(data: ETFRequest):
    ticker = data.ticker
    start_date = data.start_date
    end_date = data.end_date
    
    try:
        # Train the model if not already done (or you can skip this if already trained)
        train_model(ticker, start_date, end_date)

        # Make the prediction using the trained model
        prediction = predict_movement(ticker, start_date, end_date)
        
        # Generate plot for the ETF
        plot = generate_plot(ticker, start_date, end_date)

        return {
            "ticker": ticker,
            "predicted_return": prediction,
            "plot": plot
        }
    
    except Exception as e:
        return {"error": str(e)}

# Import FastAPI to create our server and BaseModel to structure incoming data
from fastapi import FastAPI
from pydantic import BaseModel

# Create a FastAPI app that will handle our requests
app = FastAPI()

# This is the structure of data we will receive from the frontend (your website).
class PredictionRequest(BaseModel):
    feature1: float  # This represents one type of data you will send (replace with actual features)
    feature2: float  # Another type of data (replace with actual features)

# This is an endpoint to check if the server is working.
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# This is where we receive data from the frontend (your website) and return a prediction.
@app.post("/predict")
def predict(request: PredictionRequest):
    # The prediction logic would go here (for now it's just a mock example).
    prediction = "Buy"  # Replace this with actual prediction logic
    return {"prediction": prediction}
