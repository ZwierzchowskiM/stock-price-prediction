# predict.py
import torch
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn as nn
from src.model import LSTMModel


def load_model(model_path):
    """
    Ładuje model z pliku.
    """
    print('Load model')
    model = LSTMModel(input_size=1, hidden_layer_size=50, num_layers=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def get_yahoo_data(symbol, start_date, end_date):
    """
    Pobiera dane z Yahoo Finance na podstawie symbolu akcji i dat.
    """
    print('Get data from yahoo')
    data = yf.download(symbol, start=start_date, end=end_date)
    return data


def predict_next_day(data, model, seq_length=30):
    """
    Funkcja do przewidywania ceny na następny dzień.
    """
    
    # Normalizacja danych
    scaler = MinMaxScaler()
    recent_data_scaled = scaler.fit_transform(data)  # W razie potrzeby dostosuj to do używanego scaler'a

    # Zmieniamy dane na tensor
    input_tensor = torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0)

    # Przewidywanie
    with torch.no_grad():
        prediction = model(input_tensor)

    # Inwersja normalizacji, aby uzyskać cenę w oryginalnej skali
    predicted_price = scaler.inverse_transform(prediction.detach().numpy())


    return predicted_price[0][0]
