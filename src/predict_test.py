import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_processing import load_data, clean_data, create_sequences, split_data
import os
from model import LSTMModel

# Parametry
seq_length = 30

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(os.path.abspath(__file__))  # folder src/

model_path = os.path.join(project_dir, "model/lstm_stock_model.pth")
file_path = os.path.join(project_dir, "data/^GSPC_data.csv")

# 1. Wczytanie i przygotowanie danych
data = load_data(file_path)
data = clean_data(data)
print(data.tail(10))   # Ostatnie 10 wierszy

train_data, test_data, scaler = split_data(data, test_size=0.2)
X_test, y_test = create_sequences(test_data, seq_length)

# Zmieniamy kształt danych na [samples, time_steps, features]
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Konwersja do tensora
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 3. Inicjalizacja modelu i wczytanie wag
model = LSTMModel(input_size=1, hidden_layer_size=50, num_layers=1)
model.load_state_dict(torch.load(model_path))
model.eval()

# 4. Predykcja
with torch.no_grad():
    predictions = model(X_test_tensor)

# 5. Przywracamy oryginalną skalę
predicted_prices = scaler.inverse_transform(predictions.numpy())
real_prices = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))

# 6. Wizualizacja
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label='Real')
plt.plot(predicted_prices, label='Predicted')
plt.title('Real vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()
