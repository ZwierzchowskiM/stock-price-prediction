
from src.data_processing import *

if __name__ == "__main__":
    
    file_path = "../data/^GSPC_data.csv"
    
    data = load_data(file_path)
    data = clean_data(data)
    train_data, test_data = split_data(data)
    seq_length = 30
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    print(f"Rozmiar X_train: {X_train.shape}")  
    print(f"Rozmiar y_train: {y_train.shape}") 