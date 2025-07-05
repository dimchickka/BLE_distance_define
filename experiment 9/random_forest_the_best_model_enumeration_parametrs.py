import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.utils import shuffle
from itertools import product
import matplotlib.pyplot as plt
import joblib
import os
from collections import defaultdict

# === Параметры эксперимента ===
SEQ_LENGTH_OPTIONS = [5, 10, 15]  # Длина окна
N_ESTIMATORS_OPTIONS = [50, 100, 150]  # Кол-во деревьев
MAX_DEPTH_OPTIONS = [None, 10, 20]  # Глубина деревьев (None = без ограничения)

# === Путь к обучающим данным ===
DATA_FILES = [
    ('LearningDataForLSTMFar.txt', 'Test_Far_Ready.txt'),
    # сюда можно добавить другие (имя_обучающего, имя_тестового)
]

EXPERIMENT_RESULTS = []

for data_file, test_file in DATA_FILES:
    print(f"\n🔍 Работаем с данными: {data_file}")

    df_train = pd.read_csv(data_file, header=None, names=["distance", "rssi"])
    df_test = pd.read_csv(test_file, header=None, names=["distance", "rssi"])

    for seq_len, n_trees, max_depth in product(SEQ_LENGTH_OPTIONS, N_ESTIMATORS_OPTIONS, MAX_DEPTH_OPTIONS):
        X_train, y_train = [], []
        for dist in df_train['distance'].unique():
            rssi_values = df_train[df_train['distance'] == dist]['rssi'].values
            for i in range(seq_len, len(rssi_values)):
                window = rssi_values[i - seq_len:i]
                X_train.append(window)
                y_train.append(dist)

        X_test, y_test = [], []
        for dist in df_test['distance'].unique():
            rssi_values = df_test[df_test['distance'] == dist]['rssi'].values
            for i in range(seq_len, len(rssi_values)):
                window = rssi_values[i - seq_len:i]
                X_test.append(window)
                y_test.append(dist)

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"[!] Пропускаем комбинацию (SEQ={seq_len}, trees={n_trees}, depth={max_depth}) — недостаточно данных")
            continue

        X_train, y_train = shuffle(np.array(X_train), np.array(y_train), random_state=42)
        X_test, y_test = np.array(X_test), np.array(y_test)

        model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Общий MAE
        overall_mae = mean_absolute_error(y_test, y_pred)

        # MAE по каждому расстоянию
        mae_by_distance = defaultdict(list)
        for true, pred in zip(y_test, y_pred):
            mae_by_distance[true].append(abs(true - pred))

        print(f"\n🌲 Random Forest (SEQ={seq_len}, trees={n_trees}, depth={max_depth})")
        print("📏 MAE по каждому расстоянию:")
        for dist in sorted(mae_by_distance.keys()):
            mae = np.mean(mae_by_distance[dist])
            print(f"  Расстояние {dist:.2f} м — MAE: {mae:.4f} м")

        print(f"\n📊 Общий MAE: {overall_mae:.4f} м")

        EXPERIMENT_RESULTS.append({
            "seq_len": seq_len,
            "n_trees": n_trees,
            "max_depth": max_depth if max_depth is not None else "None",
            "mae": overall_mae
        })

        # Визуализация
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="Истинное расстояние", alpha=0.7)
        plt.plot(y_pred, label="Предсказанное расстояние", alpha=0.7)
        plt.title(f"SEQ={seq_len}, trees={n_trees}, depth={max_depth} | MAE={overall_mae:.4f} м")
        plt.xlabel("Номер примера")
        plt.ylabel("Расстояние (м)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Таблица всех результатов ===
print("\n📋 Итоги эксперимента:")
for result in EXPERIMENT_RESULTS:
    print(f"SEQ={result['seq_len']}, trees={result['n_trees']}, depth={result['max_depth']} → MAE={result['mae']:.4f} м")
