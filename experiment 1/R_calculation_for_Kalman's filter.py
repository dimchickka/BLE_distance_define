import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Загрузка данных
data = np.loadtxt('../data.txt', delimiter=',')  # Формат: [расстояние, RSSI]
distance = data[:, 0]
rssi = data[:, 1]

# Словарь для хранения RSSI по расстояниям
distance_data = defaultdict(list)
for d, r in zip(distance, rssi):
    distance_data[d].append(r)

# Выбор нечётных уровней расстояний (0.1, 0.3, ..., 7.9 м)
odd_distances = np.arange(0.1, 5.0, 0.2)  # 0.1, 0.3, ..., 7.9
R_values = {}
counts = {}

for d in odd_distances:
    d = round(d, 1)  # Для точного сравнения
    if d in distance_data:
        points = distance_data[d][:100]  # Первые 100 точек
        if len(points) >= 10:  # Минимум 10 точек для статистики
            R = np.var(points)
            R_values[d] = R
            counts[d] = len(points)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(R_values.keys(), R_values.values(), 'bo-', label='Дисперсия R')
plt.xlabel('Расстояние (м)')
plt.ylabel('Дисперсия RSSI (dBm²)')
plt.title('Зависимость дисперсии шума измерений от расстояния')
plt.grid(True)
plt.legend()

# Среднее значение R
mean_R = np.mean(list(R_values.values()))
plt.axhline(mean_R, color='r', linestyle='--', 
            label=f'Среднее R = {mean_R:.2f} dBm²')
plt.legend()

plt.show()

print(f"Среднее значение R: {mean_R:.2f} (dBm)²")