import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import joblib
from collections import defaultdict

SEQ_LENGTH = 10

# === Загрузка модели ===
model = joblib.load('random_forest_model_Far.pkl')

# === Загрузка тестовых данных ===
df = pd.read_csv('Test_Far_Ready.txt', header=None, names=["distance", "rssi"])

X_test, y_test = [], []

# Формируем тестовые окна
for dist in df['distance'].unique():
    rssi_values = df[df['distance'] == dist]['rssi'].values
    for i in range(SEQ_LENGTH, len(rssi_values)):
        window = rssi_values[i-SEQ_LENGTH:i]
        X_test.append(window)
        y_test.append(dist)

X_test = np.array(X_test)
y_test = np.array(y_test)

# === Предсказание ===
y_pred = model.predict(X_test)

# === Общая ошибка ===
overall_mae = mean_absolute_error(y_test, y_pred)

# === MAE по каждому расстоянию ===
mae_by_distance = defaultdict(list)
for true, pred in zip(y_test, y_pred):
    mae_by_distance[true].append(abs(true - pred))

print("📏 MAE по каждому расстоянию:")
for dist in sorted(mae_by_distance.keys()):
    mae = np.mean(mae_by_distance[dist])
    print(f"  Расстояние {dist:.2f} м — MAE: {mae:.4f} м")

print(f"\n📊 Общий MAE: {overall_mae:.4f} м")

# === График ===
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Истинное расстояние", alpha=0.7)
plt.plot(y_pred, label="Предсказанное расстояние", alpha=0.7)
plt.title(f"Random Forest предсказание расстояния\nОбщий MAE: {overall_mae:.4f} м")
plt.xlabel("Номер примера")
plt.ylabel("Расстояние (м)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
