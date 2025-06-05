import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# === Константы фильтра Калмана ===
Q = 0.03      # Шум модели
R = 18.05     # Шум измерений (остаётся фиксированным)
P0 = 100      # Начальная неопределённость

WINDOW_SIZE = 10          # Размер скользящего окна
RESET_THRESHOLD_BASE = 3.0  # Базовый порог сброса
K_FACTOR = 20.0            # Коэффициент чувствительности к дисперсии

# === Фильтр Калмана с адаптивным порогом сброса ===
class AdaptiveKalmanFilter1D:
    def __init__(self, process_variance=Q, measurement_variance=R, initial_error=P0, reset_threshold_base=RESET_THRESHOLD_BASE):
        self.Q = process_variance
        self.R = measurement_variance
        self.P = initial_error
        self.reset_threshold_base = reset_threshold_base
        self.x = None  # Оценка состояния

    def filter(self, data):
        filtered = []

        for i, z in enumerate(data):
            if self.x is None:
                self.x = z  # Инициализация
            else:
                # Прогноз
                x_pred = self.x
                P_pred = self.P + self.Q

                # Вычисление локальной дисперсии в окне
                window_start = max(0, i - WINDOW_SIZE)
                current_window = data[window_start:i+1]
                local_var = np.var(current_window)

                # Динамический порог сброса: зависит от дисперсии
                dynamic_reset_threshold = K_FACTOR * np.sqrt(local_var)

                # Обнаружение скачка
                if abs(z - x_pred) > dynamic_reset_threshold:
                    print(f"🔄 Сброс фильтра на шаге {i}, порог = {dynamic_reset_threshold:.2f}")
                    self.x = z
                    self.P = P0
                else:
                    # Обновление
                    K = P_pred / (P_pred + self.R)
                    self.x = x_pred + K * (z - x_pred)
                    self.P = (1 - K) * P_pred

            filtered.append(self.x)
        return filtered

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df

# === Группировка и фильтрация ===
def group_and_filter(df, kalman_filter_class):
    result = defaultdict(list)
    for distance, group in df.groupby('distance'):
        rssi_values = group['rssi'].values
        kf = kalman_filter_class()
        filtered = kf.filter(rssi_values)
        result[distance] = filtered
    return result

# === Сохранение в Learning и Test файлы ===
def save_to_file(data_dict, learning_file="LearningData.txt", test_file="TestData.txt"):
    with open(learning_file, 'w') as lfile, open(test_file, 'w') as tfile:
        for distance, values in data_dict.items():
            split_idx = int(len(values) * 0.7)
            learning_part = values[:split_idx]
            test_part = values[split_idx:]

            for val in learning_part:
                lfile.write(f"{distance},{val:.2f}\n")
            for val in test_part:
                tfile.write(f"{distance},{val:.2f}\n")

    print(f"✅ Обучающие данные сохранены в {learning_file}")
    print(f"✅ Тестовые данные сохранены в {test_file}")

# === Основной запуск ===
if __name__ == "__main__":
    filename = "../data.txt"
    df = load_data(filename)

    # Фильтрация по расстояниям
    filtered_by_distance = group_and_filter(df, AdaptiveKalmanFilter1D)

    # Сохранение данных в файлы
    save_to_file(filtered_by_distance)

    # Построение графика дисперсий (для анализа)
    variances = {d: np.var(vals) for d, vals in filtered_by_distance.items()}
    distances = sorted(variances.keys())
    variance_values = [variances[d] for d in distances]

    plt.figure(figsize=(12, 6))
    plt.plot(distances, variance_values, marker='o', linestyle='-', color='blue')
    plt.title("Дисперсия RSSI после адаптивного фильтра Калмана")
    plt.xlabel("Расстояние (м)")
    plt.ylabel("Дисперсия RSSI")
    plt.grid(True)
    plt.tight_layout()
    plt.show()