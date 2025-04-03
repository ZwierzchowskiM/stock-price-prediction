import yfinance as yf
import pandas as pd
import os

# Funkcja pobierająca dane
def get_stock_data(ticker="^GSPC", start="2010-01-01", end="2024-04-01"):
    print(f"Pobieranie danych dla {ticker}...")
    data = yf.download(ticker, start=start, end=end)
    
    # Ścieżka do katalogu głównego projektu
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Tworzymy folder 'data' jeśli nie istnieje
    os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)

    # Ścieżka do zapisu pliku CSV w folderze 'data'
    file_path = os.path.join(project_dir, "data", f"{ticker}_data.csv")

    # Zapisujemy dane do pliku CSV
    data.to_csv(file_path)
    
    print(f"Dane zapisane w {file_path}")
    return data

if __name__ == "__main__":
    df = get_stock_data()
    print(df.head())