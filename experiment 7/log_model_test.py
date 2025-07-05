import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict

# === Константы ===
POINTS_PER_DISTANCE = 100
THRESHOLD_SEARCH_RANGE = np.arange(1.5, 4.05, 0.1)

# Загрузка параметров логарифмической модели
n, A = joblib.load("log_model_params.pkl")

def estimate_distance(rssi, n, A):
    return 10 ** ((A - rssi) / (10 * n))

# Загрузка данных
data = np.loadtxt("TestData.txt", delimiter=",")
true_distances_all = data[:, 0]
rssi_values_all = data[:, 1]

# Ограничение по количеству точек
grouped_data = defaultdict(list)
for dist, rssi in zip(true_distances_all, rssi_values_all):
    grouped_data[round(dist, 2)].append((dist, rssi))

filtered_pairs = []
for dist in sorted(grouped_data.keys()):
    group = grouped_data[dist]
    filtered_pairs.extend(group[:POINTS_PER_DISTANCE])

filtered_data = np.array(filtered_pairs)
true_distances = filtered_data[:, 0]
rssi_values = filtered_data[:, 1]

# Поиск оптимального порога
best_threshold = None
min_error = float('inf')

for threshold in THRESHOLD_SEARCH_RANGE:
    errors = 0
    for d, rssi in zip(true_distances, rssi_values):
        pred_dist = estimate_distance(rssi, n, A)
        predicted_near = pred_dist <= threshold
        actual_near = d <= threshold
        if predicted_near != actual_near:
            errors += 1
    error_rate = errors / len(true_distances)
    if error_rate < min_error:
        min_error = error_rate
        best_threshold = threshold

print(f"Оптимальный порог разделения: {best_threshold:.2f} м с ошибкой {min_error*100:.2f}%")

# Сохранение результатов
with open("result.txt", "w") as all_f, \
     open("Test_Near.txt", "w") as near_f, \
     open("Test_Far.txt", "w") as far_f:

    all_f.write("TrueDistance,PredictedDistance,RSSI,Class\n")

    for d, rssi in zip(true_distances, rssi_values):
        pred_dist = estimate_distance(rssi, n, A)
        classification = "Near" if pred_dist <= best_threshold else "Far"
        line = f"{d:.2f},{pred_dist:.2f},{rssi:.2f},{classification}\n"
        all_f.write(line)

        # В соответствующий файл
        if classification == "Near":
            near_f.write(f"{d:.2f},{pred_dist:.2f},{rssi:.2f}\n")
        else:
            far_f.write(f"{d:.2f},{pred_dist:.2f},{rssi:.2f}\n")

print("Результаты записаны в result.txt, Test_Near.txt и Test_Far.txt")

# === График ошибок ===
grouped_by_dist = defaultdict(list)
for dist, rssi in zip(true_distances, rssi_values):
    grouped_by_dist[round(dist, 2)].append((dist, rssi))

x_labels = []
error_rates = []

for dist in sorted(grouped_by_dist.keys()):
    group = grouped_by_dist[dist]
    group_errors = 0
    for d, rssi in group:
        pred_dist = estimate_distance(rssi, n, A)
        pred_near = pred_dist <= best_threshold
        actual_near = d <= best_threshold
        if pred_near != actual_near:
            group_errors += 1
    error_percent = 100 * group_errors / len(group)
    error_rates.append(error_percent)
    x_labels.append(dist)

plt.figure(figsize=(14, 5))
plt.bar([str(x) for x in x_labels], error_rates, color='skyblue', edgecolor='black')
plt.xticks(rotation=90)
plt.ylabel("Процент ошибок (%)")
plt.xlabel("Расстояние (м)")
plt.title(f"Ошибка классификации по расстояниям (Порог = {best_threshold:.2f} м, {POINTS_PER_DISTANCE} точек)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
