import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# === Константы фильтра Калмана ===
Q = 0.03      # Шум модели
R = 18.05     # Шум измерений
P0 = 100      # Начальная неопределённость

# === Дополнительные настройки ===
DISTANCES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9]    # Расстояния для обработки
MAX_POINTS_PER_DISTANCE = 100             # Максимум точек с каждого расстояния
WINDOW_SIZE = 10                          # Размер скользящего окна
RESET_THRESHOLD = 3.0                     # Порог изменения среднего для сброса

# === Фильтр Калмана: шаг фильтрации ===
def kalman_step(x_prev, P_prev, z, Q=Q, R=R):
    x_pred = x_prev
    P_pred = P_prev + Q

    y = z - x_pred
    S = P_pred + R
    K = P_pred / S

    x_new = x_pred + K * y
    P_new = (1 - K) * P_pred

    return x_new, P_new

# === Загрузка данных по расстояниям ===
def load_data(filename):
    data_by_distance = defaultdict(list)
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            distance = float(parts[0])
            rssi = float(parts[1])
            data_by_distance[distance].append(rssi)
    return data_by_distance

# === Фильтрация с адаптивным сбросом ===
def kalman_with_reset(rssi_values, Q=Q, R=R, P0=P0, window_size=20, threshold=3.0):
    filtered = []
    reset_points = []

    x = rssi_values[0]
    P = P0
    prev_mean = x

    window = [x]

    for i, z in enumerate(rssi_values):
        x, P = kalman_step(x, P, z, Q, R)
        filtered.append(x)

        # Добавляем в окно
        window.append(z)
        if len(window) > window_size:
            window.pop(0)

        # Если окно заполнено — сравниваем средние
        if len(window) == window_size:
            current_mean = np.mean(window)
            mean_diff = abs(current_mean - prev_mean)

            if mean_diff > threshold:
                print(f"🔄 Сброс фильтра на шаге {i}, изменение среднего: {mean_diff:.2f} дБ")
                x = current_mean
                P = P0
                prev_mean = current_mean
                reset_points.append(i)

    return filtered, reset_points

# === Сохранение отфильтрованных данных в файл ===
def save_filtered_data(filtered_rssi, distances_list, points_per_distance, filename="filteredData.txt"):
    with open(filename, 'w') as f:
        idx = 0
        for dist in distances_list:
            count = points_per_distance[idx]
            for i in range(count):
                rssi_val = filtered_rssi[i + sum(points_per_distance[:idx])]
                f.write(f"{dist},{rssi_val:.2f}\n")
            idx += 1
    print(f"✅ Отфильтрованные данные сохранены в {filename}")

# === Визуализация результатов ===
def plot_results(original, filtered, reset_points, distances_list, points_per_distance):
    plt.figure(figsize=(16, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(distances_list)))

    # Оригинал и отфильтрованный сигнал
    plt.plot(original, label='Оригинальный RSSI', alpha=0.7, marker='o', markersize=3)
    plt.plot(filtered, label='Отфильтрованный RSSI', linewidth=2)

    # Вертикальные линии для разделения участков
    cumulative = 0
    for idx, dist in enumerate(distances_list):
        cumulative += points_per_distance[idx]
        plt.axvline(x=cumulative, color='gray', linestyle='--', alpha=0.5)
        plt.text(cumulative + 2, min(original), f"{dist} м", rotation=90, va='bottom', fontsize=10)

    # Точки сброса
    for rp in reset_points:
        plt.axvline(x=rp, color='red', linestyle='--', alpha=0.5, linewidth=1)

    plt.title("Фильтрация RSSI при движении между несколькими расстояниями\n(Q=%.2f, R=%.2f, P0=%.2f)" % (Q, R, P0))
    plt.xlabel('Временной шаг')
    plt.ylabel('RSSI (dBm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Основной запуск ===
if __name__ == "__main__":
    filename = "../data.txt"
    output_file = "filteredDataLearning.txt"
    data_by_distance = load_data(filename)

    combined_rssi = []
    points_per_distance = []
    distances_list = []

    for dist in DISTANCES:
        dist_key = round(dist, 2)
        if dist_key not in data_by_distance:
            print(f"Нет данных для расстояния {dist_key} м")
            continue

        samples = data_by_distance[dist_key][:MAX_POINTS_PER_DISTANCE]
        combined_rssi.extend(samples)
        points_per_distance.append(len(samples))
        distances_list.append(dist_key)

    print(f"Обработано точек: {len(combined_rssi)}")
    print("Распределение по расстояниям:", points_per_distance)

    # === Фильтрация с адаптивным сбросом ===
    filtered_rssi, reset_indices = kalman_with_reset(
        combined_rssi,
        Q=Q,
        R=R,
        P0=P0,
        window_size=WINDOW_SIZE,
        threshold=RESET_THRESHOLD
    )

    # === Сохранение результатов в файл ===
    save_filtered_data(filtered_rssi, distances_list, points_per_distance, output_file)

    # === Визуализация ===
    plot_results(combined_rssi, filtered_rssi, reset_indices, distances_list, points_per_distance)