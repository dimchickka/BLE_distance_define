import torch
import torch.nn as nn
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# --- Константы ---
N_TREES = 3
TREE_MODEL_PATH_TEMPLATE = "weak_experts/tree_expert_{}.pkl"

LSTM_MODEL_PATH_TEMPLATE = "../experiment 8/lstm_distance_model_{}.pth"
SCALER_X_PATH_TEMPLATE = "../experiment 8/scaler_x_{}.pkl"
SCALER_Y_PATH_TEMPLATE = "../experiment 8/scaler_y_{}.pkl"

TEST_DATA_PATH = "TestData.txt"

# --- Модель LSTM (как в train_lstm_Near.py) ---
class DistanceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(DistanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- Функция загрузки деревьев ---
def load_trees(n_trees=N_TREES):
    trees = []
    for i in range(1, n_trees + 1):  # начиная с 1, если имена с 1
        path = TREE_MODEL_PATH_TEMPLATE.format(i)
        with open(path, "rb") as f:
            tree = pickle.load(f)
            trees.append(tree)
    print(f"[✓] Загружено {len(trees)} деревьев.")
    return trees

# --- Функция загрузки LSTM и скейлеров ---
def load_lstm_and_scalers(prefix):
    model_path = LSTM_MODEL_PATH_TEMPLATE.format(prefix)
    scaler_x_path = SCALER_X_PATH_TEMPLATE.format(prefix)
    scaler_y_path = SCALER_Y_PATH_TEMPLATE.format(prefix)

    model = DistanceLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    return model, scaler_x, scaler_y

# --- Загрузка тестовых данных ---
def load_test_data(filepath):
    # Формат файла: true_distance,RSSI (одна точка в строке)
    data = np.loadtxt(filepath, delimiter=',')
    return data[:, 0], data[:, 1]

# --- Основная функция ---
def main():
    # Загружаем деревья
    trees = load_trees(N_TREES)

    # Загружаем модели и скейлеры
    model_near, scaler_x_near, scaler_y_near = load_lstm_and_scalers("Near")
    model_far, scaler_x_far, scaler_y_far = load_lstm_and_scalers("Far")

    # Загружаем тестовые данные
    true_distances, rssi_values = load_test_data(TEST_DATA_PATH)

    mse_errors = []
    predictions_all = []

    for i in range(len(rssi_values)):
        rssi = rssi_values[i]
        true_dist = true_distances[i]

        # Предсказания деревьев
        votes = []
        # Для классификации деревья ожидают вход (RSSI). Предположим, что деревья обучены на простых признаках RSSI.
        # Если деревья ожидают вектор, нужно сделать reshape.
        for tree in trees:
            pred = tree.predict([[rssi]])[0]
            votes.append(pred)

        # Выбираем кластер по большинству голосов
        cluster = Counter(votes).most_common(1)[0][0]

        # Подготавливаем данные для LSTM (последовательность из одинаковых RSSI, тк у нас нет полной последовательности)
        seq = np.array([rssi] * 10).reshape(1, 10, 1)  # размер (batch=1, seq_len=10, features=1)

        if cluster == 0:
            # Near
            X_scaled = scaler_x_near.transform(seq.reshape(-1, 1)).reshape(1, 10, 1)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model_near(X_tensor).item()
            pred_dist = scaler_y_near.inverse_transform([[pred_scaled]])[0][0]

        else:
            # Far
            X_scaled = scaler_x_far.transform(seq.reshape(-1, 1)).reshape(1, 10, 1)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model_far(X_tensor).item()
            pred_dist = scaler_y_far.inverse_transform([[pred_scaled]])[0][0]

        predictions_all.append(pred_dist)
        mse_errors.append((true_dist - pred_dist) ** 2)

    mse = np.mean(mse_errors)
    print(f"Среднеквадратичная ошибка (MSE) по тесту: {mse:.4f}")

    # --- График ошибки по каждой точке ---
    errors_percent = np.abs(np.array(true_distances) - np.array(predictions_all)) / np.array(true_distances) * 100

    plt.figure(figsize=(14, 6))
    plt.plot(errors_percent, marker='.', linestyle='-', color='blue')
    plt.title("Процент ошибки по каждой тестовой точке")
    plt.xlabel("Индекс точки")
    plt.ylabel("Процент ошибки (%)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
