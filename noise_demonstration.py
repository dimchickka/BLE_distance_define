import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Константы фильтра Калмана ===
Q = 0.03      # Шум модели
R = 18.05     # Шум измерений
P0 = 100      # Начальная неопределённость

WINDOW_SIZE = 10
RESET_THRESHOLD = 3.0  # Порог сброса фильтра

# === Фильтр Калмана с возможностью сброса ===
class KalmanFilter1D:
    def __init__(self, process_variance=Q, measurement_variance=R, initial_error=P0, reset_threshold=RESET_THRESHOLD):
        self.Q = process_variance
        self.R = measurement_variance
        self.P = initial_error
        self.reset_threshold = reset_threshold
        self.x = None  # Оценка (state estimate)

    def filter(self, data):
        filtered = []

        for z in data:
            if self.x is None:
                self.x = z  # инициализация
            else:
                # Предсказание
                x_pred = self.x
                P_pred = self.P + self.Q

                # Обнаружение скачка
                if abs(z - x_pred) > self.reset_threshold:
                    # Сброс фильтра
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
df = pd.read_csv("data.txt", header=None, names=['distance', 'rssi'])

# === Группировка и фильтрация ===
variances = {}

for distance, group in df.groupby('distance'):
    rssi_values = group['rssi'].values

    # Применение фильтра Калмана с возможностью сброса
    kf = KalmanFilter1D()
    filtered_rssi = kf.filter(rssi_values)

    # Расчёт дисперсии отфильтрованных данных
    var = np.var(filtered_rssi)
    variances[distance] = var

# === Построение графика дисперсий ===
distances = sorted(variances.keys())
variance_values = [variances[d] for d in distances]

plt.figure(figsize=(12, 6))
plt.plot(distances, variance_values, marker='o', linestyle='-', color='blue')
plt.title("Дисперсия RSSI после фильтра Калмана с детектором скачков")
plt.xlabel("Расстояние (м)")
plt.ylabel("Дисперсия RSSI")
plt.grid(True)
plt.tight_layout()
plt.show()
