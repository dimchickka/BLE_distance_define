# test_lstm.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

class DistanceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(DistanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Берём последнее состояние
        return out
# === Гиперпараметры ===
SEQ_LENGTH = 10
MODEL_PATH = "lstm_distance_model_Far.pth"
SCALER_X_PATH = "scaler_x_Far.pkl"
SCALER_Y_PATH = "scaler_y_Far.pkl"

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df['rssi'].values, df['distance'].values

# === Создание последовательностей (по группам расстояния) ===
def create_sequences_grouped_by_distance(rssi, distance, seq_length=10):
    X, y = [], []
    rssi = np.array(rssi)
    distance = np.array(distance)
    unique_distances = np.unique(distance)

    for d in unique_distances:
        rssi_group = rssi[distance == d]

        for i in range(seq_length, len(rssi_group)):
            X.append(rssi_group[i - seq_length:i])
            y.append(d)  # одно расстояние на всю последовательность
    return np.array(X), np.array(y)

# === Предсказание расстояния одной последовательности ===
def predict(model, scaler_X, scaler_y, rssi_sequence):
    X_scaled = scaler_X.transform(np.array(rssi_sequence).reshape(-1, 1))
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, 1)
    with torch.no_grad():
        prediction = model(tensor)
    return scaler_y.inverse_transform(prediction.numpy().reshape(-1, 1))[0][0]

# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка тестовых данных ---
    rssi_values, true_distances = load_data("Test_Far_Ready.txt")

    # --- Шаг 2: Формирование последовательностей ---
    X_test, y_test = create_sequences_grouped_by_distance(rssi_values, true_distances, SEQ_LENGTH)

    # --- Шаг 3: Загрузка модели и нормализаторов ---
    model = DistanceLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # --- Шаг 4: Предсказания ---
    predictions = []
    for i in range(len(X_test)):
        sequence = X_test[i]
        predicted_distance = predict(model, scaler_X, scaler_y, sequence)
        predictions.append(predicted_distance)

    predictions = np.array(predictions)
    true_distances_aligned = y_test  # Уже правильно выровненные значения

    print(f"Количество истинных значений: {len(true_distances_aligned)}")
    print(f"Количество предсказаний: {len(predictions)}")

    # === Оценка ===
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(true_distances_aligned, predictions)
    mae = mean_absolute_error(true_distances_aligned, predictions)

    print(f"\nMSE: {mse:.4f} м²")
    print(f"MAE: {mae:.4f} м")

    # === График ===
    plt.figure(figsize=(14, 6))
    plt.plot(true_distances_aligned, label='Истинное расстояние', alpha=0.8, linewidth=2)
    plt.plot(predictions, label='Предсказанное LSTM', alpha=0.95, linewidth=2)
    plt.title(f"Сравнение истинного и предсказанного расстояния\nMSE: {mse:.4f} м², MAE: {mae:.4f} м")
    plt.xlabel("Порядковый номер последовательности")
    plt.ylabel("Расстояние (м)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
