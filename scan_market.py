import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import json
from datetime import datetime

model = load_model("stock_lstm_model.h5")
scaler = joblib.load("scaler.gz")

sequence_length = 60

# Load ticker list
tickers = pd.read_csv("nifty500.csv")["Symbol"].tolist()
tickers = [t + ".NS" for t in tickers]

results = []

for symbol in tickers:
    try:
        data = yf.download(symbol, period="2y", progress=False)
        close_prices = data["Close"].values.reshape(-1, 1)
        scaled = scaler.transform(close_prices)
        last_60 = scaled[-sequence_length:]
        input_data = np.reshape(last_60, (1, sequence_length, 1))
        pred_scaled = model.predict(input_data)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

        results.append({
            "symbol": symbol,
            "current_price": float(close_prices[-1][0]),
            "predicted_price": round(float(predicted_price), 2)
        })
    except:
        continue

json.dump(results, open("market_predictions.json", "w"), indent=2)

print(f"Scan complete at {datetime.now()} — {len(results)} stocks processed.")
