
from src.data_processing import *
from src.data_loader import *
from src.predict import *

base_dir = os.path.dirname(os.path.abspath(__file__))  # folder src/
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Ścieżka do modelu
    model_path = os.path.join(base_dir, "model/lstm_stock_model.pth")
    

    # Załaduj model
    model = load_model(model_path)

    # Pobierz dane z Yahoo Finance
    data = get_stock_data()
    data = clean_data(data)

    # Używamy tylko ceny zamknięcia ('Close')
    data_close = data[['Close']]

    # Predykcja ceny na następny dzień
    predicted_price = predict_next_day(data_close, model)
    print(f"Predykcja ceny na następny dzień: {predicted_price}")

if __name__ == "__main__":
    main()

