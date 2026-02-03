# 📊 Crop Price Prediction Dataset Explanation

## 1. Dataset Overview
- **Dataset Source:** AGMARKNET (Agricultural Marketing Information Network), Govt. of India.
- **Nature of Data:** Time-Series Data (Sequential data points indexed by time).
- **Domain:** Agricultural Market Prices (Mandi Prices).
- **Goal:** To predict future prices (`Modal_Price`) based on historical trends.

---

## 2. Dataset Structure (Columns)

The dataset contains daily records of crop prices in various markets (Mandis).

| Column Name | Full Name | Type | Description |
|---|---|---|---|
| **Price Date** | Date | Date/Time | The specific date when the price was recorded. This is our *Time Index*. |
| **State** | State Name | Categorical | The state where the Mandi is located (e.g., Maharashtra). |
| **District** | District Name | Categorical | The district of the Mandi. |
| **Market** | Mandi Name | Categorical | The specific market (e.g., Azadpur, Pune, Mumbai). Prices vary by market. |
| **Commodity** | Crop Name | Categorical | The crop being sold (e.g., Tomato, Onion, Wheat). |
| **Min_Price** | Minimum Price | Numerical | The lowest price at which the crop was sold that day (₹/Quintal). |
| **Max_Price** | Maximum Price | Numerical | The highest price at which the crop was sold that day (₹/Quintal). |
| **Modal_Price** | Average Price | Numerical | **TARGET VARIABLE.** The most common or average trading price for the day. This is what we predict. |

---

## 3. Why Historical Data is Needed?
In Time-Series Forecasting, **"History repeats itself."**
- **Seasonality:** Onion prices rise every year during monsoon/festivals.
- **Trends:** General inflation causes prices to go up over 5-10 years.
- **Cycles:** Agricultural cycles (sowing/harvesting) affect supply and price.

By analyzing the past 3-5 years of data, our models (ARIMA/LSTM) learn these patterns to forecast the future.

---

## 4. Handling Missing Values & Outliers
Real-world data is messy. Before feeding it to the model, we:
1.  **Imputation:** Fill missing dates using *Forward Fill* (assuming price didn't change) or *Interpolation* (average of previous and next day).
2.  **Outlier Removal:** Sometimes data entry errors show Tomato price as ₹10,000/Quintal instead of ₹1,000. We remove these spikes to prevent confusing the model.

---

## 5. Why is this Dataset Suitable?
1.  **Granularity:** It has *daily* data points, which is perfect for short-term prediction (next 7-14 days).
2.  **Volume:** Thousands of records allow Deep Learning models (LSTM) to learn complex patterns.
3.  **Stationarity Potential:** Although prices change, the *patterns* (like seasonal spikes) are consistent, making it solvable with ARIMA/LSTM.
