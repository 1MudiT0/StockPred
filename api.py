from flask import Flask, request, jsonify
import yfinance as yf
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler (make sure these files are in the same folder as this api.py)
model = load_model("stock_lstm_model.h5")
scaler = joblib.load("scaler.gz")

sequence_length = 60


# -------------------------------------------------------
# Single Stock Prediction
# -------------------------------------------------------
@app.route("/predict", methods=["GET"])
def predict():
    ticker = request.args.get("symbol", "AAPL")

    # Download recent data
    data = yf.download(ticker, period="2y", progress=False)

    if data.empty:
        return jsonify({"error": "Invalid symbol or no data found"}), 400

    close_prices = data["Close"].values.reshape(-1, 1)
    current_price = float(close_prices[-1][0])

    # Scale data
    scaled_data = scaler.transform(close_prices)

    # Last 60 days as input
    last_60 = scaled_data[-sequence_length:]
    input_data = np.reshape(last_60, (1, sequence_length, 1))

    # Predict next close price
    scaled_prediction = model.predict(input_data)
    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]

    # Percent change and trend
    percent_change = ((predicted_price - current_price) / current_price) * 100
    trend = "UP" if predicted_price > current_price else "DOWN"

    return jsonify({
        "symbol": ticker,
        "current_price": round(current_price, 2),
        "predicted_price": round(float(predicted_price), 2),
        "percent_change": round(float(percent_change), 2),
        "trend": trend
    })


# -------------------------------------------------------
# Full Market Predictions (NIFTY 500 results)
# -------------------------------------------------------
@app.route("/market", methods=["GET"])
def market():
    try:
        data = json.load(open("market_predictions.json"))
        return jsonify(data)
    except:
        return jsonify({"error": "Run scan_market.py first to generate market_predictions.json"}), 500


# -------------------------------------------------------
# Run Server
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
