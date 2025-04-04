import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def get_stock_data_to_CVS(ticker="^GSPC", start="2010-01-01"):
    end = datetime.today().strftime('%Y-%m-%d')
    print(f"Pobieranie danych dla {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
    file_path = os.path.join(project_dir, "data", f"{ticker}_data.csv")

   
    data.to_csv(file_path)
    
    print(f"Dane zapisane w {file_path}")
    return data

def get_stock_data():
    print ('Get stock data')
    ticker="^GSPC"
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    print("End date:", end)
    print("Start date:", start)
    print(f"Pobieranie danych dla {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    return data

def visualize_stock_data(file_path):
    
    data = pd.read_csv(file_path)
    print(data.columns)

    print(data.head())
    data = data.iloc[2:].reset_index(drop=True)
    print(data.head())
    
    if 'Close' not in data.columns:
        print(f"Brak kolumny 'Close' w pliku {file_path}.")
        return
    

   
    plt.figure(figsize=(12, 6))
    plt.plot(data['Price'], data['Close'], label='Cena zamknięcia')
    plt.legend()
    plt.title('Wykres ceny zamknięcia')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    get_stock_data()
    file_path = "../data/^GSPC_data.csv"
    #check_data(file_path)
    #visualize_stock_data(file_path)