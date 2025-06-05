# test_lstm.py
from sklearn.metrics import mean_absolute_error
import collections
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from train_lstm import DistanceLSTM  # Архитектура модели

# === Гиперпараметры ===
SEQ_LENGTH = 10      # Длина последовательности
MODEL_PATH = "lstm_distance_model.pth"
SCALER_X_PATH = "scaler_x.pkl"
SCALER_Y_PATH = "scaler_y.pkl"

# === Загрузка тестовых данных ===
def load_test_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df['rssi'].values, df['distance'].values

# === Создание последовательностей внутри одного расстояния ===
def create_sequences_grouped_by_distance(rssi_values, distance_values, seq_length=10):
    X, y = [], []

    unique_distances = np.unique(distance_values)

    for dist in unique_distances:
        mask = (distance_values == dist)
        rssi_group = rssi_values[mask]
        distance_group = distance_values[mask]

        for i in range(seq_length, len(rssi_group)):
            X.append(rssi_group[i-seq_length:i])
            y.append(distance_group[i])

    return np.array(X), np.array(y)

# === Предсказание с нормализацией ===
def predict(model, scaler_X, scaler_y, sequence):
    X_scaled = scaler_X.transform(sequence.reshape(-1, 1))
    tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # batch_size = 1
    with torch.no_grad():
        prediction = model(tensor)
    return scaler_y.inverse_transform(prediction.numpy().reshape(-1, 1))[0][0]

# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка тестовых данных ---
    rssi_values, true_distances = load_test_data("TestData.txt")

    # --- Шаг 2: Формирование последовательностей ---
    X_test, y_test = create_sequences_grouped_by_distance(rssi_values, true_distances, SEQ_LENGTH)

    print(f"Сформировано тестовых последовательностей: {len(X_test)}")

    # --- Шаг 3: Загрузка обученной модели и скалеров ---
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
    true_distances_aligned = y_test  # Уже совпадают по длине

    print(f"Количество истинных значений: {len(true_distances_aligned)}")
    print(f"Количество предсказаний: {len(predictions)}")

    # === Оценка точности по каждому расстоянию ===
    # Группируем по расстояниям
    distance_to_true_pred = collections.defaultdict(lambda: {"true": [], "pred": []})

    for true_val, pred_val in zip(true_distances_aligned, predictions):
        distance_to_true_pred[round(true_val, 2)]["true"].append(true_val)
        distance_to_true_pred[round(true_val, 2)]["pred"].append(pred_val)

    # Считаем MAE по каждому расстоянию
    print("\n📏 MAE по каждому расстоянию:")
    all_mae = []
    for dist in sorted(distance_to_true_pred.keys()):
        true_vals = distance_to_true_pred[dist]["true"]
        pred_vals = distance_to_true_pred[dist]["pred"]
        mae = mean_absolute_error(true_vals, pred_vals)
        all_mae.append(mae)
        print(f"  Расстояние {dist:.2f} м — MAE: {mae:.4f} м")

    # Общая средняя MAE (не взвешенная)
    avg_mae = np.mean(all_mae)
    print(f"\n📊 Среднее значение MAE по всем расстояниям: {avg_mae:.4f} м")

    # === График ===
    plt.figure(figsize=(14, 6))
    plt.plot(true_distances_aligned, label='Истинное расстояние', alpha=0.8, linewidth=2)
    plt.plot(predictions, label='Предсказанное LSTM', alpha=0.95, linewidth=2)
    plt.title("Предсказание расстояния через LSTM")
    plt.xlabel("Номер последовательности")
    plt.ylabel("Расстояние (м)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()