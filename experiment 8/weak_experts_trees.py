import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Конфигурация
NUM_EXPERTS = 5  # Увеличим количество экспертов
SAVE_DIR = "weak_experts"
DIAG_DIR = "diagnostics"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)


# Загрузка данных
def load_data():
    cluster0 = np.loadtxt("Cluster0.txt", delimiter=",")
    cluster1 = np.loadtxt("Cluster1.txt", delimiter=",")
    X0 = cluster0[:, 1].reshape(-1, 1)
    X1 = cluster1[:, 1].reshape(-1, 1)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    return X, y, X0, X1


X, y, X0, X1 = load_data()

# Анализ данных перед обучением
plt.figure(figsize=(10, 6))
plt.hist(X0, bins=30, alpha=0.5, label='Cluster 0 (Near)', density=True)
plt.hist(X1, bins=30, alpha=0.5, label='Cluster 1 (Far)', density=True)
plt.title("Нормализованное распределение RSSI")
plt.xlabel("RSSI (dBm)")
plt.ylabel("Плотность")
plt.legend()
plt.savefig(f"{DIAG_DIR}/data_distribution.png")
plt.close()

print(f"Общее количество точек: {len(X)}")
print(f"Среднее RSSI Cluster 0: {np.mean(X0):.2f} ± {np.std(X0):.2f}")
print(f"Среднее RSSI Cluster 1: {np.mean(X1):.2f} ± {np.std(X1):.2f}")

# Создаем разнообразных слабых экспертов
for i in range(NUM_EXPERTS):
    # 1. Уникальная стратегия выборки для каждого эксперта
    if i % 3 == 0:
        # Стратегия 1: обычный бутстрэп
        X_sample, y_sample = resample(X, y, replace=True, random_state=i)
    elif i % 3 == 1:
        # Стратегия 2: undersampling majority class
        n_samples = min(len(X0), len(X1))
        idx0 = np.random.choice(len(X0), n_samples, replace=False)
        idx1 = np.random.choice(len(X1), n_samples, replace=False)
        X_sample = np.vstack([X0[idx0], X1[idx1]])
        y_sample = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    else:
        # Стратегия 3: взвешенная выборка
        weights = np.where(y == 0, 1 / len(X0), 1 / len(X1))
        idx = np.random.choice(len(X), len(X) // 2, p=weights / weights.sum())
        X_sample, y_sample = X[idx], y[idx]

    # 2. Уникальные параметры для каждого дерева
    params = {
        'max_depth': np.random.randint(2, 6),
        'min_samples_split': np.random.randint(2, 10),
        'max_features': np.random.choice([None, 'sqrt', 0.5, 0.3]),
        'class_weight': np.random.choice([None, 'balanced', {0: 1, 1: 2}]),
        'random_state': i * 100
    }

    # 3. Создаем и обучаем дерево
    tree = DecisionTreeClassifier(**params)
    tree.fit(X_sample, y_sample)

    # 4. Сохраняем модель
    joblib.dump(tree, f"{SAVE_DIR}/tree_expert_{i + 1}.pkl")

    # 5. Диагностика
    train_acc = accuracy_score(y_sample, tree.predict(X_sample))
    test_acc = accuracy_score(y, tree.predict(X))

    print(f"\nЭксперт #{i + 1}:")
    print(f"Стратегия выборки: {'Бутстрэп' if i % 3 == 0 else 'Undersample' if i % 3 == 1 else 'Weighted'}")
    print(f"Параметры: {params}")
    print(f"Размер обучающей выборки: {len(X_sample)}")
    print(f"Accuracy (train/test): {train_acc:.2f}/{test_acc:.2f}")

    # Сохраняем примеры предсказаний
    test_rssi = np.linspace(min(X.min(), -100), max(X.max(), -20), 5)
    preds = tree.predict(test_rssi.reshape(-1, 1))
    print(f"Тестовые предсказания для RSSI {test_rssi.round(1)}: {preds}")

# Создаем файл с метаинформацией
with open(f"{DIAG_DIR}/experts_info.txt", "w") as f:
    f.write("Метаинформация о слабых экспертах:\n")
    f.write(f"Всего экспертов: {NUM_EXPERTS}\n")
    f.write(f"Общее количество точек: {len(X)}\n")
    f.write(f"Распределение классов: {len(X0)} Near, {len(X1)} Far\n")