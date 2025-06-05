import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# === Пути к файлам ===
MODEL_PATH = "svm_distance_model.pkl"
SCALER_PATH = "svm_scaler.pkl"

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    X = df['rssi'].values.reshape(-1, 1)
    y = df['distance'].values
    return X, y, df['distance'].unique()

# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка тестовых данных ---
    print("Загрузка тестовых данных...")
    X_test, y_true, unique_distances = load_data("TestData.txt")

    # --- Шаг 2: Загрузка обученной SVM и скалера ---
    print("Загрузка модели и скалера...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # --- Шаг 3: Предсказание расстояния ---
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    # --- Шаг 4: Общая ошибка ---
    mse = mean_squared_error(y_true, predictions)
    mae_total = mean_absolute_error(y_true, predictions)
    print(f"\n🧪 Общая ошибка:")
    print(f"   MSE = {mse:.4f} м²")
    print(f"   MAE = {mae_total:.4f} м\n")

    # --- Шаг 5: Ошибка по каждому расстоянию ---
    distance_wise_mae = {}

    for d in sorted(unique_distances):
        mask = (y_true == d)
        y_true_d = y_true[mask]
        pred_d = predictions[mask]

        if len(y_true_d) > 0:
            mae = mean_absolute_error(y_true_d, pred_d)
            distance_wise_mae[d] = mae

    # --- Шаг 6: Вывод MAE по расстояниям ---
    print("🧪 MAE по каждому расстоянию:")
    for dist, mae in distance_wise_mae.items():
        print(f"{dist:.2f} м: MAE = {mae:.4f} м")

    # --- Шаг 7: График истинных и предсказанных значений ---
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label='Истинное расстояние', linewidth=2)
    plt.plot(predictions, label='Предсказанное SVM', linewidth=2)
    plt.title(f"Оценка расстояния через SVM\nMSE: {mse:.4f} м², MAE: {mae_total:.4f} м")
    plt.xlabel("Тестовая точка")
    plt.ylabel("Расстояние (м)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()