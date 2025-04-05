# 📈 LSTM Stock Price Predictor

This project uses a Long Short-Term Memory (LSTM) neural network to predict the next day's stock price based on historical closing prices.


## 📦 Tech Stack

- **Python 3.8+**
- **PyTorch** – deep learning framework
- **pandas** – data manipulation
- **scikit-learn** – preprocessing
- **yfinance** – downloading stock data
- **NumPy** – numerical operations
- **matplotlib** – visualization



## 📁 Project Structure

```plaintext
.
├── data/                   # Input CSV files (e.g., ^GSPC_data.csv)
├── src/
│   └── model.py           # LSTM model class definition
├── data_processing.py     # Data cleaning and sequence preparation
├── train.py               # Model training script
├── predict.py             # Next-day prediction script
├── lstm_stock_model.pth   # Saved trained model
└── README.md              # Project documentation

```

## ⚙️ Model Configuration
sequence length: 30 days
hidden size: 50
num LSTM layers: 2
batch size: 32
epochs: 20
learning rate: 0.001


## ✅ Notes
The model uses only the closing price (no volume, open, high, or low data).
At least 30 days of historical data is required to make a prediction.
It's optimized for stable indices like S&P 500 (^GSPC), but can be adapted for individual stocks.