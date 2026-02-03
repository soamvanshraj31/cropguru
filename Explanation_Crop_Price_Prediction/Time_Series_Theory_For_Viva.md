# 📉 Time-Series Forecasting Theory (For Viva)

## 1. What is Time-Series Data?
Time-series data is a sequence of data points collected at regular time intervals.
*   **Key Feature:** The order matters! (You cannot shuffle the data like in regular regression).
*   **Example:** Daily stock prices, Hourly temperature, Monthly rainfall.

---

## 2. Components of a Time Series
Every time series can be broken down into 3 parts:
1.  **Trend:** Long-term direction (Upward or Downward).
    *   *Example:* Inflation causes prices to go up over 10 years.
2.  **Seasonality:** Repeating patterns at fixed intervals.
    *   *Example:* Mango prices drop every summer (Supply increases).
3.  **Residual (Noise):** Random fluctuations that cannot be explained.

---

## 3. What is Stationarity? (Crucial Concept)
A time series is **Stationary** if its statistical properties (Mean, Variance) do not change over time.
*   **Why it matters:** ARIMA *only* works on stationary data.
*   **Test:** We use the **Augmented Dickey-Fuller (ADF)** test.
*   **How to fix Non-Stationarity:** Use **Differencing** (Subtract Today's value - Yesterday's value).

---

## 4. ARIMA Model Explained
**ARIMA** stands for **Auto-Regressive Integrated Moving Average**. It has 3 parameters `(p, d, q)`:

1.  **AR (Auto-Regressive) `p`:**
    *   Uses past values to predict future.
    *   *Logic:* "If it rained yesterday, it's likely to rain today."
2.  **I (Integrated) `d`:**
    *   The number of times we differenced the data to make it stationary.
3.  **MA (Moving Average) `q`:**
    *   Uses past forecast errors to correct future predictions.

---

## 5. LSTM Model Explained
**LSTM** stands for **Long Short-Term Memory**.
*   **Problem with normal Neural Networks:** They "forget" early data in a long sequence.
*   **LSTM Solution:** It has a special "Memory Cell" that can store information for a long time.

**How LSTM works (Simple Analogy):**
Imagine reading a book.
*   You remember the main character's name from Chapter 1 (Long-term memory).
*   You read the current sentence (Short-term input).
*   LSTM decides:
    1.  **Forget Gate:** What irrelevant info to throw away?
    2.  **Input Gate:** What new info to store?
    3.  **Output Gate:** What to predict next?

---

## 6. Comparison: ARIMA vs. LSTM

| Feature | ARIMA | LSTM |
|---|---|---|
| **Type** | Statistical Model | Deep Learning Model |
| **Data Size** | Works well with small data | Needs LARGE data to perform well |
| **Complexity** | Captures Linear patterns | Captures Complex / Non-Linear patterns |
| **Speed** | Fast to train | Slow to train (Computationally heavy) |
| **Best For** | Short-term forecasting | Long-term trends & complex data |

**Conclusion for Viva:**
"I used a Hybrid approach. ARIMA is great for capturing the seasonal baseline, while LSTM captures complex market fluctuations. Comparing both ensures the best accuracy."
