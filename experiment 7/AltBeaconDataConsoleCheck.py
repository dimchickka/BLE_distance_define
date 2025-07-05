import numpy as np
import pickle


# === Модель AltBeacon ===
def altbeacon_model(rssi, alpha, beta, gamma, r0):
    return alpha * (rssi / r0) ** beta + gamma


# === Загрузка обученной модели ===
def load_model(filename="altbeacon_model.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# === Загрузка обучающих данных ===
def load_learning_data(filename="LearningData.txt"):
    return np.loadtxt(filename, delimiter=",")


# === Основной скрипт ===
def save_predictions_to_file():
    try:
        # Загрузка модели и данных
        model = load_model()
        data = load_learning_data()

        true_distances = data[:, 0]
        rssi_values = data[:, 1]

        predicted = altbeacon_model(
            rssi_values,
            model["alpha"],
            model["beta"],
            model["gamma"],
            model["r0"]
        )

        # Запись в файл
        with open("resultAltBeacon.txt", "w") as f:
            for true_d, pred_d in zip(true_distances, predicted):
                f.write(f"Истинное расстояние: {true_d:.2f} м | Предсказанное: {pred_d:.2f} м\n")

        print("✅ Результаты сохранены в resultAltBeacon.txt")

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Проверь файл Learning.txt и модель altbeacon_model.pkl")


save_predictions_to_file()
