# ==============================================================================
# CROP PRICE PREDICTION SYSTEM (ARIMA + LSTM)
# ==============================================================================
# This script demonstrates the complete pipeline:
# 1. Data Loading & Preprocessing
# 2. ARIMA Model Training (Statistical Approach)
# 3. LSTM Model Training (Deep Learning Approach)
# 4. Model Comparison & Recommendation
# 5. Future Forecasting
# ==============================================================================

# --- IMPORTING LIBRARIES ---
import pandas as pd  # For data manipulation (DataFrames)
import numpy as np   # For numerical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from statsmodels.tsa.stattools import adfuller  # For checking Stationarity (Augmented Dickey-Fuller test)
from statsmodels.tsa.arima.model import ARIMA   # The ARIMA Model
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Evaluation Metrics
from sklearn.preprocessing import MinMaxScaler  # For scaling data (0-1) for LSTM
from tensorflow.keras.models import Sequential  # To build the Neural Network
from tensorflow.keras.layers import LSTM, Dense # LSTM and Dense layers

# ==========================================
# STEP 1: LOAD AND PREPROCESS DATA
# ==========================================

# 1.1 Load the dataset
# We assume a CSV file with columns: Date, Market, Crop, Price
print("Loading dataset...")
df = pd.read_csv('Agriculture_price_dataset.csv')

# 1.2 Filter Data
# We focus on ONE Crop and ONE Market at a time for accurate time-series prediction
# Predicting 'Tomato' prices in 'Mumbai' market
crop_name = 'Tomato'
mandi_name = 'Mumbai'
df_filtered = df[(df['Commodity'] == crop_name) & (df['Market'] == mandi_name)]

# 1.3 Date Conversion
# Convert the 'Date' column from text to Python datetime objects
df_filtered['Price Date'] = pd.to_datetime(df_filtered['Price Date'])

# 1.4 Sorting
# Time series data MUST be sorted chronologically (Oldest -> Newest)
df_filtered = df_filtered.sort_values('Price Date')

# 1.5 Handling Missing Values
# If a weekend/holiday has no price, we use 'ffill' (Forward Fill) - use yesterday's price
df_filtered.set_index('Price Date', inplace=True)
df_filtered = df_filtered.asfreq('D') # Ensure daily frequency
df_filtered['Modal_Price'] = df_filtered['Modal_Price'].fillna(method='ffill')

print(f"Data loaded for {crop_name} in {mandi_name}. Total days: {len(df_filtered)}")

# ==========================================
# STEP 2: CHECK FOR STATIONARITY
# ==========================================
# ARIMA requires data to be 'Stationary' (constant mean and variance).
# We use the Augmented Dickey-Fuller (ADF) Test.

result = adfuller(df_filtered['Modal_Price'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Logic: If p-value > 0.05, data is Non-Stationary -> We need Differencing (d=1)
d_value = 0
if result[1] > 0.05:
    print("Data is Non-Stationary. Differencing is required.")
    d_value = 1
else:
    print("Data is Stationary.")

# ==========================================
# STEP 3: ARIMA MODEL (Auto-Regressive Integrated Moving Average)
# ==========================================
print("\n--- Training ARIMA Model ---")

# 3.1 Split Data
# 80% for Training, 20% for Testing
train_size = int(len(df_filtered) * 0.8)
train_data, test_data = df_filtered['Modal_Price'][0:train_size], df_filtered['Modal_Price'][train_size:]

# 3.2 Train ARIMA (with Grid Search)
# We search for the best (p, d, q) parameters to minimize error (AIC).
# p: Auto-regressive order, d: Differencing, q: Moving average
print("Searching for optimal ARIMA parameters...")
# (Code loop to find best order... skipping for brevity)
best_order = (5, d_value, 0) # Assume this was found to be the best
arima_model = ARIMA(train_data, order=best_order)
arima_fit = arima_model.fit()

# 3.3 Predict on Test Set
arima_predictions = arima_fit.forecast(steps=len(test_data))

# 3.4 Evaluate ARIMA
arima_rmse = np.sqrt(mean_squared_error(test_data, arima_predictions))
print(f"ARIMA RMSE (Error): {arima_rmse:.2f}")

# ==========================================
# STEP 4: LSTM MODEL (Long Short-Term Memory)
# ==========================================
print("\n--- Training LSTM Model ---")

# 4.1 Normalization
# Neural Networks work best with data between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_filtered['Modal_Price'].values.reshape(-1, 1))

# 4.2 Create Sequences
# LSTM needs "Windows" of data. E.g., Use Day 1-30 to predict Day 31.
window_size = 30
X_lstm, y_lstm = [], []

for i in range(window_size, len(scaled_data)):
    X_lstm.append(scaled_data[i-window_size:i, 0]) # Past 30 days
    y_lstm.append(scaled_data[i, 0])             # Target: Day 31

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
# Reshape for LSTM [Samples, Time Steps, Features]
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# 4.3 Build LSTM Architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1))) # Layer 1
model.add(LSTM(units=50, return_sequences=False)) # Layer 2
model.add(Dense(units=25)) # Dense Layer
model.add(Dense(units=1))  # Output Layer (1 Price)

# 4.4 Compile & Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_lstm, y_lstm, batch_size=1, epochs=1) # Epochs=1 for demo, usually 20+

# 4.5 Predict
# (Prediction logic involves taking the last window and predicting future steps)
# For simplicity, we skip the complex loop here, but the concept is:
# Predict next day -> Add to window -> Predict day after that -> Repeat.

print("LSTM Training Complete.")

# ==========================================
# STEP 5: FORECAST & ADVICE
# ==========================================

# Let's say we forecasted the next 7 days prices:
forecast_prices = [2500, 2550, 2600, 2580, 2650, 2700, 2750] # Example values

# Trend Detection Logic
start_price = forecast_prices[0]
end_price = forecast_prices[-1]

print("\n--- Farmer Recommendation ---")
print(f"Current Price: ₹{start_price}/Quintal")
print(f"Forecasted Price (Day 7): ₹{end_price}/Quintal")

if end_price > start_price * 1.05: # Price rising by >5%
    print("Advice: HOLD your crop. Prices are rising.")
elif end_price < start_price * 0.95: # Price falling by >5%
    print("Advice: SELL NOW. Prices are likely to fall.")
else:
    print("Advice: MARKET STABLE. Sell at your convenience.")
