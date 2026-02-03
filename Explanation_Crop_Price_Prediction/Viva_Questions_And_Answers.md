# 🎓 Viva Questions & Answers: Crop Price Prediction

Read these questions until you are comfortable answering them naturally.

---

## 🟢 Basic Concepts

**Q1: What is the goal of the Price Prediction module?**
**Answer:** To forecast future crop prices based on historical data, helping farmers decide the best time to sell their produce for maximum profit.

**Q2: Which algorithms did you use?**
**Answer:** I implemented both **ARIMA** (a statistical model) and **LSTM** (a deep learning model) and compared their performance to select the best one.

**Q3: What is Time-Series Forecasting?**
**Answer:** It is a technique to predict future values based on previously observed values, where the data is collected at regular time intervals (like daily prices).

**Q4: What is the source of your dataset?**
**Answer:** The data mimics real-world records from **AGMARKNET** (Government of India), containing daily modal prices for various crops and mandis.

**Q5: What is "Modal Price"?**
**Answer:** It is the most common or average price at which the crop was traded in the market on a specific day. It is more stable than Min or Max price.

---

## 🟡 Technical Questions (ARIMA)

**Q6: What does ARIMA stand for?**
**Answer:** Auto-Regressive Integrated Moving Average.

**Q7: Explain the parameters (p, d, q).**
**Answer:**
- **p (Auto-Regressive):** Dependence on past values.
- **d (Integrated):** Number of times we differenced the data to make it stationary.
- **q (Moving Average):** Dependence on past forecast errors.

**Q8: What is Stationarity and why is it important?**
**Answer:** A time series is stationary if its mean and variance are constant over time. ARIMA requires stationarity to work correctly. If data is trending up/down, it's non-stationary.

**Q9: How do you check for Stationarity?**
**Answer:** I used the **Augmented Dickey-Fuller (ADF)** test. If the p-value is < 0.05, the data is stationary. If not, I apply differencing.

---

## 🟠 Technical Questions (LSTM)

**Q10: Why did you use LSTM?**
**Answer:** LSTM is excellent at learning long-term dependencies and non-linear patterns in data, which simple statistical models like ARIMA might miss.

**Q11: How does LSTM handle sequential data?**
**Answer:** LSTM processes data step-by-step and maintains a "cell state" (memory) that carries relevant information throughout the sequence, allowing it to remember patterns from long ago.

**Q12: Why do you scale data for LSTM?**
**Answer:** Neural networks converge faster and perform better when input values are small (usually 0 to 1). I used `MinMaxScaler` for this.

**Q13: What is the "Window Size" in your LSTM?**
**Answer:** The Window Size (e.g., 30 days) is the number of past days the model looks at to predict the price for the next day.

---

## 🔴 Evaluation & Advanced

**Q14: How did you compare ARIMA and LSTM?**
**Answer:** I used **RMSE (Root Mean Squared Error)** and **MAPE (Mean Absolute Percentage Error)**. The model with the lower error on the test set was selected.

**Q15: Which model performed better?**
**Answer:** (Customize based on your results, but usually:) "It varied by crop, but generally, LSTM performed better for crops with complex price fluctuations, while ARIMA was sufficient for stable trends."

**Q16: What is Overfitting?**
**Answer:** When a model learns the training data (and its noise) too well but fails on new data. I prevented this in LSTM by using **Dropout layers**.

**Q17: How does the "Farmer Recommendation" work?**
**Answer:** I compare the forecasted price with the current price. If the forecast shows a significant rise (>5%), the system advises the farmer to "HOLD". If it drops, it advises "SELL".

**Q18: How do you handle missing data?**
**Answer:** I used **Forward Fill (ffill)**, which propagates the last valid price forward. This assumes prices remain stable on holidays/weekends.

**Q19: What are the limitations of your system?**
**Answer:** It relies on historical patterns. It cannot predict sudden external shocks like a new government export ban or an unexpected natural disaster.

**Q20: How can this be improved?**
**Answer:** By adding "Exogenous variables" like Rainfall data, Transport costs, and International prices to the model (ARIMAX or Multivariate LSTM).

**Q21: What is the 'Vanishing Gradient' problem?**
**Answer:** In normal RNNs, gradients become too small during training, making it hard to learn long-term patterns. LSTM solves this using its unique gating mechanism.

**Q22: Why did you split data chronologically and not randomly?**
**Answer:** Because it's Time-Series data! The order matters. We must train on the *past* and test on the *future*. Random splitting would involve "cheating" by using future data to predict the past.

**Q23: How far into the future can you predict?**
**Answer:** The accuracy decreases as we predict further out. My system is optimized for short-term forecasting (next 7 to 14 days), which is most useful for immediate selling decisions.

**Q24: Is your model deployed?**
**Answer:** Yes, the trained models are saved as `.pkl` or `.h5` files and loaded by a Flask backend to serve real-time predictions to the React frontend.
