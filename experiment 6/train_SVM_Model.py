import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# === Настройки ===
MODEL_PATH = "svm_distance_model.pkl"
SCALER_PATH = "svm_scaler.pkl"

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    X = df['rssi'].values.reshape(-1, 1)
    y = df['distance'].values
    return X, y

# === Основной запуск ===
if __name__ == "__main__":
    # --- Шаг 1: Загрузка данных ---
    print("Загрузка данных...")
    X_train, y_train = load_data("LearningData.txt")

    # --- Шаг 2: Нормализация данных ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # --- Шаг 3: Обучение SVM с RBF-ядром ---
    print("Обучение SVM с RBF-ядром...")
    model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    model.fit(X_train_scaled, y_train)

    # --- Шаг 4: Сохранение модели и скалера ---
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Модель сохранена: {os.path.basename(MODEL_PATH)}")
    print(f"✅ Скалер сохранён: {os.path.basename(SCALER_PATH)}")