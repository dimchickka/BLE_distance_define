import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

# === Настройки ===
POLY_DEGREE = 2         # Степень полинома
MODEL_PATH = "poly_distance_model.pkl"  # Путь для сохранения модели

# === Загрузка данных ===
def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['distance', 'rssi'])
    return df['rssi'].values.reshape(-1, 1), df['distance'].values

# === Основной запуск ===
if __name__ == "__main__":
    X_train, y_train = load_data("LearningData.txt")

    # === Создание полиномиальных признаков ===
    poly = PolynomialFeatures(degree=POLY_DEGREE)
    X_poly_train = poly.fit_transform(X_train)

    # === Обучение модели ===
    print("Обучение полиномиальной регрессии...")
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X_poly_train, y_train)

    # === Сохранение модели ===
    joblib.dump(model, MODEL_PATH)
    joblib.dump(poly, "poly_transformer.pkl")
    print(f"✅ Модель сохранена: {os.path.basename(MODEL_PATH)}")
    print(f"✅ Трансформер сохранен: poly_transformer.pkl")