import numpy as np
import pickle
import matplotlib.pyplot as plt

def altbeacon_model(rssi, alpha, beta, gamma, r0):
    # rssi, r0 — в dBm, модель как задано
    # Предсказывает расстояние
    return alpha * (rssi / r0) ** beta + gamma

def load_test_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data[:,0], data[:,1]  # true_distances, rssi_values

def load_model(filename='altbeacon_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def classify(true_distances, pred_distances, threshold):
    true_labels = (true_distances <= threshold).astype(int)
    pred_labels = (pred_distances <= threshold).astype(int)
    return true_labels, pred_labels

def test_model():
    true_distances, rssi_values = load_test_data('TestData.txt')
    model = load_model()

    # Предсказания расстояний по AltBeacon модели
    distances_pred = altbeacon_model(rssi_values, model['alpha'], model['beta'], model['gamma'], model['r0'])

    # Кандидатные пороги: уникальные истинные расстояния в диапазоне [1, 3] м
    candidate_thresholds = np.unique(true_distances[(true_distances >= 1.0) & (true_distances <= 3.0)])

    errors = []
    for th in candidate_thresholds:
        true_labels, pred_labels = classify(true_distances, distances_pred, th)
        err = np.mean(true_labels != pred_labels)
        errors.append(err)

    errors = np.array(errors)
    best_idx = np.argmin(errors)
    best_threshold = candidate_thresholds[best_idx]
    best_error = errors[best_idx]

    print(f"Оптимальный порог классификации: {best_threshold:.2f} м с ошибкой {best_error*100:.2f}%")

    # Подсчёт ошибки классификации по истинным расстояниям для визуализации
    unique_dist = np.unique(true_distances)
    error_rates = []
    for d in unique_dist:
        mask = true_distances == d
        if np.sum(mask) == 0:
            error_rates.append(0)
            continue
        # Классификация с выбранным порогом best_threshold
        true_lab = (true_distances[mask] <= best_threshold).astype(int)
        pred_lab = (distances_pred[mask] <= best_threshold).astype(int)
        error_rate = np.mean(true_lab != pred_lab)
        error_rates.append(error_rate * 100)  # в процентах

    # Построение графика ошибок по расстоянию
    plt.figure(figsize=(10,5))
    plt.plot(unique_dist, error_rates, marker='o', color='red')
    plt.xlabel('Истинное расстояние (м)')
    plt.ylabel('Ошибка классификации (%)')
    plt.title(f'Ошибки классификации по расстоянию (порог={best_threshold:.2f} м)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_model()
