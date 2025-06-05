import numpy as np
from collections import defaultdict

# === Константы фильтра Калмана ===
Q = 0.03      # Шум модели
R = 18.05     # Шум измерений
P0 = 100      # Начальная неопределённость

WINDOW_SIZE = 10
RESET_THRESHOLD = 3.0

# === Фильтр Калмана ===
def kalman_step(x_prev, P_prev, z, Q=Q, R=R):
    x_pred = x_prev
    P_pred = P_prev + Q
    y = z - x_pred
    S = P_pred + R
    K = P_pred / S
    x_new = x_pred + K * y
    P_new = (1 - K) * P_pred
    return x_new, P_new

# === Фильтрация с адаптивным сбросом ===
def kalman_with_reset(rssi_values, Q=Q, R=R, P0=P0, window_size=WINDOW_SIZE, threshold=RESET_THRESHOLD):
    filtered = []
    x = rssi_values[0]
    P = P0
    prev_mean = x
    window = [x]
    for z in rssi_values:
        x, P = kalman_step(x, P, z, Q, R)
        filtered.append(x)
        window.append(z)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            current_mean = np.mean(window)
            if abs(current_mean - prev_mean) > threshold:
                x = current_mean
                P = P0
                prev_mean = current_mean
    return filtered

# === Загрузка данных ===
def load_data(filename):
    data_by_distance = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                dist, rssi = map(float, line.strip().split(','))
                data_by_distance[round(dist, 2)].append(rssi)
    return data_by_distance

# === Сохранение данных ===
def save_data(data_by_distance, filename):
    with open(filename, 'w') as f:
        for dist, rssi_list in data_by_distance.items():
            for rssi in rssi_list:
                f.write(f"{dist},{rssi:.2f}\n")
    print(f"✅ Сохранено: {filename}")

# === Основной запуск ===
if __name__ == "__main__":
    data = load_data("data.txt")

    learning_data = defaultdict(list)
    test_data = defaultdict(list)

    for dist, rssi_values in data.items():
        if len(rssi_values) < 2:
            continue
        filtered = kalman_with_reset(rssi_values)

        n = len(filtered)
        n_train = int(n * 0.7)

        learning_data[dist] = filtered[:n_train]
        test_data[dist] = filtered[n_train:]

    save_data(learning_data, "LearningData.txt")
    save_data(test_data, "TestData.txt")
