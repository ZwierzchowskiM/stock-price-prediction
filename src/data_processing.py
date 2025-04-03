import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data):

    data.rename(columns={'Price': 'Date'}, inplace=True)
    data = data.iloc[2:].reset_index(drop=True)
    
    return data

def split_data(data, test_size=0.2):
    """
    Dzieli dane na zbiór treningowy i testowy.
    """
    X = data[['Open', 'High', 'Low', 'Volume']]  
    y = data['Close']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test

def scale_data(data):
    """
    Normalizuje dane (skaluje) do zakresu [0, 1] przy użyciu MinMaxScaler.
    """
    scaler = MinMaxScaler()
    data[['Close']] = scaler.fit_transform(data[['Close']])
    data[['High']] = scaler.fit_transform(data[['High']])
    data[['Low']] = scaler.fit_transform(data[['Low']])
    data[['Open']] = scaler.fit_transform(data[['Open']])
    data[['OpeVolumen']] = scaler.fit_transform(data[['Volume']])
    return data
    print("Dane zostały znormalizowane.")

def load_data(file_path):
    """
    Wczytuje dane z pliku CSV.
    """
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  
    return data

if __name__ == "__main__":
    
    file_path = "../data/^GSPC_data.csv"
    
    data = load_data(file_path)
    data = prepare_data(data)
    data = scale_data(data)
    print(data.head())
