import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# === Параметры ===
SEQ_LENGTH = 10

# === Загрузка данных ===
df = pd.read_csv('LearningDataForLSTMNear.txt', header=None, names=["distance", "rssi"])

X, y = [], []

# Группируем по каждому расстоянию
for dist in df['distance'].unique():
    rssi_values = df[df['distance'] == dist]['rssi'].values
    for i in range(SEQ_LENGTH, len(rssi_values)):
        window = rssi_values[i-SEQ_LENGTH:i]
        X.append(window)
        y.append(dist)

X = np.array(X)
y = np.array(y)

print(f"Обучающих примеров: {len(X)}")

# === Обучение модели ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# === Сохранение модели ===
joblib.dump(model, 'random_forest_model_Near.pkl')
print("✅ Модель сохранена")
