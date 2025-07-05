import numpy as np
from scipy.optimize import curve_fit
import pickle

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    distances = data[:, 0]
    rssi = data[:, 1]
    return distances, rssi

def altbeacon_model(rssi, alpha, beta, gamma, r0):
    # Формула: d = alpha * (rssi / r0)^beta + gamma
    return alpha * (rssi / r0) ** beta + gamma

def fit_model(distances, rssi):
    # Находим r0 — среднее RSSI на расстоянии около 1 м
    mask_1m = (distances >= 0.95) & (distances <= 1.05)
    if np.sum(mask_1m) == 0:
        raise ValueError("Нет данных для расстояния ~1м!")
    r0 = np.mean(rssi[mask_1m])
    print(f"r0 (RSSI на 1 м): {r0:.2f} dBm")

    # Подгоняем параметры alpha, beta, gamma
    params, _ = curve_fit(
        lambda r, alpha, beta, gamma: altbeacon_model(r, alpha, beta, gamma, r0),
        rssi, distances,
        p0=[1.0, -2.0, 0.0],
        bounds=([0, -10, -5], [10, 10, 5])
    )

    alpha, beta, gamma = params
    print(f"Коэффициенты: alpha={alpha:.4f}, beta={beta:.4f}, gamma={gamma:.4f}")

    model = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'r0': r0}
    with open('altbeacon_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

if __name__ == "__main__":
    try:
        distances, rssi = load_data('LearningData.txt')
        fit_model(distances, rssi)
    except Exception as e:
        print(f"Ошибка: {e}")
