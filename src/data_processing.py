import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """
    Wczytuje dane z pliku CSV.
    """
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()  
    return data


def clean_data(data):

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


def split_data(data, test_size=0.2):

    
    # Skalowanie danych (zakres 0-1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Close']])

    # Podział na zbiór treningowy i testowy (np. 80% train, 20% test)
    split_index = int(len(data_scaled) * (1 - test_size))
    train_data = data_scaled[:split_index]
    test_data = data_scaled[split_index:]

    return train_data, test_data, scaler  # scaler będzie potrzebny do odwrotnej transformacji

def create_sequences(data, seq_length=30):
    """
    Tworzy sekwencje czasowe dla LSTM.
    
    :param data: Dane wejściowe (Pandas DataFrame, Series lub NumPy array).
    :param seq_length: Liczba kroków czasowych w jednej sekwencji.
    :return: X (wejściowe sekwencje), y (wartości do przewidzenia)
    """
    X, y = [], []

    # Jeśli to DataFrame lub Series, konwertujemy na NumPy array
    if hasattr(data, 'values'):
        data = data.values

    # Upewniamy się, że dane są w formacie 2D (n_samples, n_features)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


