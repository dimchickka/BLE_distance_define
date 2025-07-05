import numpy as np
from scipy.optimize import curve_fit
import joblib

# Загрузка данных с указанием правильного разделителя
data = np.loadtxt("LearningData.txt", delimiter=",")  # <-- здесь фикс
distances = data[:, 0]
rssi_values = data[:, 1]

# Физическая логарифмическая модель
def log_model(d, n, A):
    return -10 * n * np.log10(d) + A

# Убираем нулевые расстояния
mask = distances > 0
distances = distances[mask]
rssi_values = rssi_values[mask]

# Обучение модели
params, _ = curve_fit(log_model, distances, rssi_values, p0=[2.0, -40])
n_opt, A_opt = params
print(f"Оптимальные параметры: n = {n_opt:.4f}, A = {A_opt:.4f}")

# Сохранение модели
joblib.dump((n_opt, A_opt), "log_model_params.pkl")
