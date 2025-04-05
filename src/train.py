import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from data_processing import *
import joblib
import os

# Parametry
seq_length = 30
test_size = 0.2
epochs = 20
batch_size = 32
learning_rate = 0.001

base_dir = os.path.dirname(os.path.abspath(__file__))  # folder src/
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(project_dir, "data/^GSPC_data.csv")

# Ładowanie i przygotowanie danych
data = load_data(file_path)
data = clean_data(data)
train_data, test_data, scaler = split_data(data)
seq_length = 30
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Tworzenie sekwencji
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Zmieniamy kształt danych na [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Przekształcamy dane na tensory
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Tworzymy DataLoader do łatwiejszego zarządzania danymi
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Tworzymy model LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        # LSTM warstwa
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Weźmy ostatni wynik LSTM (h_n) i przekażmy go do warstwy Dense
        out = self.fc(lstm_out[:, -1, :])
        return out

# Parametry modelu
input_size = 1  # Tylko cena zamknięcia
hidden_layer_size = 50
output_size = 1  # Prognozujemy tylko jedną wartość (cenę zamknięcia)

# Inicjalizacja modelu
model = LSTMModel(input_size, hidden_layer_size, output_size)

# Strata i optymalizator
criterion = nn.MSELoss()  # Funkcja straty dla regresji
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Trenowanie modelu
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        # Wyczyść gradienty
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X_batch)

        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_batch shape: {y_batch.shape}")

        # Oblicz stratę
        loss = criterion(y_pred, y_batch)

        # Backward pass
        loss.backward()

        # Zaktualizuj parametry
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoka {epoch + 1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

# Zapisz model
torch.save(model.state_dict(), "lstm_stock_model.pth")

# Prognozowanie na zbiorze testowym
model.eval()  # Zmieniamy model na tryb ewaluacji
with torch.no_grad():
    predictions = model(X_test_tensor)

# Przekształcamy prognozy z powrotem na oryginalną skalę
predictions = predictions.numpy()
predictions = scaler.inverse_transform(predictions)

# Zapisz wytrenowany scaler do pliku
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")

print("Predykcje: ", predictions[:5])  # Wyświetlenie pierwszych 5 prognoz
