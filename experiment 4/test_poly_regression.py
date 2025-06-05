import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Настройки ===
POLY_DEGREE = 2         # Полином 2-й степени
MODEL_PATH = "poly_distance_model.pkl"
SCALER_PATH = "poly_transformer.pkl"

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df['distance'].values, df['distance'].values, df['rssi'].values

# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка тестовых данных ---
    _, distances_true, rssi_values = load_data("TestData.txt")
    print(f"Загружено {len(rssi_values)} тестовых точек")

    # --- Шаг 2: Загрузка модели и трансформера ---
    model = joblib.load(MODEL_PATH)
    poly = joblib.load(SCALER_PATH)

    # --- Шаг 3: Подготовка входных данных для регрессии ---
    X_test = rssi_values.reshape(-1, 1)
    X_poly_test = poly.transform(X_test)

    # --- Шаг 4: Предсказание расстояния ---
    predictions = model.predict(X_poly_test)

    # --- Шаг 5: Вычисление общей ошибки ---
    mse = mean_squared_error(distances_true, predictions)
    mae_total = mean_absolute_error(distances_true, predictions)

    print(f"\n🧪 Общая ошибка на всём наборе:")
    print(f"   MSE = {mse:.4f} м²")
    print(f"   MAE = {mae_total:.4f} м\n")

    # --- Шаг 6: Вычисление MAE по каждому расстоянию отдельно ---
    unique_distances = np.unique(distances_true)
    distance_wise_mae = {}

    for d in unique_distances:
        mask = (distances_true == d)
        y_true_by_d = distances_true[mask]
        y_pred_by_d = predictions[mask]
        mae_by_d = mean_absolute_error(y_true_by_d, y_pred_by_d)
        distance_wise_mae[d] = mae_by_d

    # --- Шаг 7: Вывод MAE по расстояниям ---
    print("🧪 MAE по каждому расстоянию:")
    for dist, mae in sorted(distance_wise_mae.items()):
        print(f"{dist:.2f} м: MAE = {mae:.4f} м")

    # --- Шаг 8: График истинного и предсказанного расстояния ---
    plt.figure(figsize=(14, 6))
    plt.plot(distances_true, label='Истинное расстояние', alpha=0.8, linewidth=2)
    plt.plot(predictions, label='Предсказанное полиномиальной регрессией', alpha=0.95, linewidth=2)

    plt.title(f"Полиномиальная регрессия 2-й степени\nMSE: {mse:.4f} м², MAE: {mae_total:.4f} м")
    plt.xlabel("Тестовая точка")
    plt.ylabel("Расстояние (м)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()