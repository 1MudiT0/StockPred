import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# 1. Download Data
ticker = "AAPL"   # <-- You can change this to any stock symbol
data = yf.download(ticker, period="5y")
close_prices = data["Close"].values.reshape(-1, 1)

# 2. Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 3. Create training sequences
X = []
y = []
sequence_length = 60  # use last 60 days to predict next day

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# 4. Reshape for LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 5. Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

# 6. Train the model
model.fit(X, y, epochs=10, batch_size=32)

# 7. Save the model
model.save("stock_lstm_model.h5")

print("Model training complete & saved as stock_lstm_model.h5")

import joblib
joblib.dump(scaler,"scaler.gz")
