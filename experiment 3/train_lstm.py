# train_lstm_Near.py

from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import os

# === Настройки ===
SEQ_LENGTH = 10       # Длина последовательности (количество точек RSSI)
HIDDEN_SIZE = 64      # Размер скрытого слоя LSTM
NUM_LAYERS = 2        # Число слоёв в LSTM
EPOCHS = 100          # Количество эпох обучения
BATCH_SIZE = 16       # Размер батча
LEARNING_RATE = 0.001  # Скорость обучения
MODEL_PATH = "lstm_distance_model.pth"
SCALER_X_PATH = "scaler_x.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

# === Загрузка данных из файла ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df['rssi'].values, df['distance'].values


# === Формирование последовательностей внутри одного расстояния ===
def create_sequences_grouped_by_distance(rssi_values, distance_values, seq_length=10):
    X, y = [], []

    unique_distances = np.unique(distance_values)

    for dist in unique_distances:
        indices = (distance_values == dist)
        rssi_group = rssi_values[indices]

        # Создаём последовательности только внутри одного расстояния
        for i in range(seq_length, len(rssi_group)):
            sequence = rssi_group[i - seq_length:i]
            X.append(sequence)
            y.append(dist)  # Все точки этого участка соответствуют одному расстоянию

    return np.array(X), np.array(y)


# === LSTM модель ===
class DistanceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(DistanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Берём последнее состояние
        return out


# === Нормализация данных ===
def normalize_data(X, y):
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(-1, SEQ_LENGTH, 1)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    return X_scaled, y_scaled, scaler_X, scaler_y


# === Сохранение модели и скалеров ===
def save_model(model, scaler_X, scaler_y):
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)
    print(f"✅ Модель сохранена: {os.path.basename(MODEL_PATH)}")
    print(f"✅ Скалеры сохранены: {os.path.basename(SCALER_X_PATH)}, {os.path.basename(SCALER_Y_PATH)}")


# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка данных ---
    rssi_values, distances = load_data("LearningData.txt")

    # --- Шаг 2: Формирование последовательностей ---
    X, y = create_sequences_grouped_by_distance(rssi_values, distances, SEQ_LENGTH)
    print(f"Сформировано последовательностей: {X.shape[0]}")

    # --- Шаг 3: Перемешивание данных между расстояниями ---
    X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

    # --- Шаг 4: Нормализация ---
    X_scaled, y_scaled, scaler_X, scaler_y = normalize_data(X_shuffled, y_shuffled)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).squeeze()

    # --- Шаг 5: Разделение выборки (можно убрать shuffle) ---
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.15, shuffle=False)

    # --- Шаг 6: Инициализация модели ---
    model = DistanceLSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Шаг 7: Обучение ---
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_X, batch_y = X_train[indices], y_train[indices]

            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # --- Шаг 8: График ошибки по эпохам ---
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Ошибка обучения LSTM\n(MSE Loss)")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()

    # --- Шаг 9: Сохранение модели и скалеров ---
    save_model(model, scaler_X, scaler_y)