import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from collections import defaultdict
from sklearn.utils import shuffle
import joblib

# === Константы ===
SEQ_LENGTH = 5
N_ESTIMATORS = 150
MAX_DEPTH = 20
DATA_PATH = 'LearningDataForLSTMFar.txt'
TEST_SPLIT = 0.2
VARIANTS = ["classic", "global_shuffle", "by_distance_shuffle", "stride_window", "random_chunks"]
RANDOM_SEED = 42

# === Загрузка данных ===
df = pd.read_csv(DATA_PATH, header=None, names=["distance", "rssi"])

def create_dataset(df, method="classic"):
    X, y = [], []
    if method == "classic":
        for dist in df['distance'].unique():
            rssi_values = df[df['distance'] == dist]['rssi'].values
            for i in range(SEQ_LENGTH, len(rssi_values)):
                X.append(rssi_values[i - SEQ_LENGTH:i])
                y.append(dist)

    elif method == "global_shuffle":
        df_shuffled = shuffle(df, random_state=RANDOM_SEED)
        rssi_values = df_shuffled['rssi'].values
        distances = df_shuffled['distance'].values
        for i in range(SEQ_LENGTH, len(rssi_values)):
            X.append(rssi_values[i - SEQ_LENGTH:i])
            y.append(distances[i])

    elif method == "by_distance_shuffle":
        for dist in df['distance'].unique():
            group = df[df['distance'] == dist]
            group = shuffle(group, random_state=RANDOM_SEED)
            rssi_values = group['rssi'].values
            for i in range(SEQ_LENGTH, len(rssi_values)):
                X.append(rssi_values[i - SEQ_LENGTH:i])
                y.append(dist)

    elif method == "stride_window":
        for dist in df['distance'].unique():
            rssi_values = df[df['distance'] == dist]['rssi'].values
            for i in range(0, len(rssi_values) - SEQ_LENGTH, 2):  # шаг 2
                X.append(rssi_values[i:i + SEQ_LENGTH])
                y.append(dist)

    elif method == "random_chunks":
        all_rssi = df['rssi'].values
        all_dist = df['distance'].values
        for _ in range(4000):  # ограничим число окон
            start = np.random.randint(0, len(all_rssi) - SEQ_LENGTH)
            X.append(all_rssi[start:start + SEQ_LENGTH])
            y.append(all_dist[start + SEQ_LENGTH - 1])

    else:
        raise ValueError(f"Неизвестный метод формирования: {method}")

    return np.array(X), np.array(y)


# === Основной цикл по методам ===
results = []
for variant in VARIANTS:
    print(f"\n🧪 Вариант: {variant}")
    X, y = create_dataset(df, method=variant)

    # Делим на обучение и тест
    split_index = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Обучение
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    # Предсказание
    y_pred = model.predict(X_test)

    # MAE общий
    overall_mae = mean_absolute_error(y_test, y_pred)

    # MAE по расстояниям
    mae_by_distance = defaultdict(list)
    for true, pred in zip(y_test, y_pred):
        mae_by_distance[true].append(abs(true - pred))

    print(f"📊 Общий MAE: {overall_mae:.4f} м")
    results.append((variant, overall_mae))


# === Таблица результатов ===
print("\n📋 Сводка по всем вариантам:")
for variant, mae in results:
    print(f"{variant:20s} → MAE = {mae:.4f} м")
