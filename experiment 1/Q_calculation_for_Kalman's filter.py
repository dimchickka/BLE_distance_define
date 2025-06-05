import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Загрузка данных
data = np.loadtxt('../data.txt', delimiter=',')  # Колонки: расстояние, RSSI
distance = data[:, 0]
rssi = data[:, 1]

# Группировка данных по расстояниям
distance_data = defaultdict(list)
for d, r in zip(distance, rssi):
    distance_data[round(d, 1)].append(r)  # Округление до 0.1 м

# Выбор нечётных расстояний (0.1, 0.3, ..., 4.9 м)
odd_distances = np.arange(0.1, 5.0, 0.2)
Q_values = {}
delta_t = 0.5  # Время между измерениями (10 Гц)

for d in odd_distances:
    d = round(d, 1)
    if d in distance_data:
        points = np.array(distance_data[d][:100])  # Первые 100 точек
        if len(points) >= 10:
            # Разности между соседними измерениями
            diffs = np.diff(points[::5])
            # Дисперсия разностей (σ²_q)
            sigma_sq = np.var(diffs, ddof=1)
            # Шум процесса Q
            Q = sigma_sq * delta_t
            Q_values[d] = Q

# Построение графика Q
plt.figure(figsize=(10, 6))
plt.plot(Q_values.keys(), Q_values.values(), 'ro-', label='$Q$ (шум процесса)')
plt.xlabel('Расстояние (м)')
plt.ylabel('$Q$ (dBm²)')
plt.title('Зависимость шума процесса $Q$ от расстояния')
plt.grid(True)
plt.legend()

# Среднее Q
mean_Q = np.mean(list(Q_values.values()))
plt.axhline(mean_Q, color='black', linestyle='--',
            label=f'Среднее $Q$ = {mean_Q:.4f} dBm²')
plt.legend()
plt.show()