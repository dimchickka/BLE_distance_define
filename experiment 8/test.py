import numpy as np
import torch
import joblib  # Используем joblib вместо pickle для совместимости
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

# Константы
N_TREES = 5
WEAK_EXPERTS_DIR = "weak_experts"
LSTM_DIR = "../experiment 8"
TEST_DATA_PATH = "TestData.txt"


# LSTM модель должна точно соответствовать обученной модели
class DistanceLSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def load_trees(n_trees=3, load_dir="weak_experts"):
    trees = []
    for i in range(1, n_trees + 1):
        path = os.path.join(load_dir, f"tree_expert_{i}.pkl")
        tree = joblib.load(path)
        trees.append(tree)
    print(f"[✓] Загружено {len(trees)} деревьев.")
    return trees


def load_lstm_and_scalers(prefix):
    model_path = os.path.join(LSTM_DIR, f"lstm_distance_model_{prefix}.pth")
    scaler_x_path = os.path.join(LSTM_DIR, f"scaler_x_{prefix}.pkl")
    scaler_y_path = os.path.join(LSTM_DIR, f"scaler_y_{prefix}.pkl")

    # Проверка существования файлов
    if not all(os.path.exists(p) for p in [model_path, scaler_x_path, scaler_y_path]):
        missing = [p for p in [model_path, scaler_x_path, scaler_y_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Отсутствуют файлы: {missing}")

    # Загрузка модели
    model = DistanceLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Загрузка скейлеров с помощью joblib
    try:
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
    except Exception as e:
        raise ValueError(f"Ошибка загрузки скейлера: {str(e)}. Файл может быть поврежден.")

    return model, scaler_x, scaler_y


def predict_distance_lstm(model, scaler_x, scaler_y, rssi):
    # Подготавливаем данные
    rssi_scaled = scaler_x.transform(np.array(rssi).reshape(-1, 1))
    input_tensor = torch.tensor(rssi_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()

    pred = scaler_y.inverse_transform(pred_scaled)
    return pred.flatten()[0]


def majority_vote(trees, rssi_point):
    votes = []
    for tree in trees:
        pred = tree.predict([[rssi_point]])
        votes.append(pred[0])
    return max(set(votes), key=votes.count)


def main():
    # Проверка существования тестовых данных
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Тестовые данные не найдены: {TEST_DATA_PATH}")

    # Загружаем данные
    data = np.loadtxt(TEST_DATA_PATH, delimiter=",")
    true_distances = data[:, 0]
    rssi_values = data[:, 1]

    # Загружаем слабых экспертов
    try:
        trees = load_trees(N_TREES)
    except Exception as e:
        print(f"Ошибка загрузки деревьев: {str(e)}")
        return

    # Загружаем LSTM модели
    try:
        model_near, scaler_x_near, scaler_y_near = load_lstm_and_scalers("Near")
        model_far, scaler_x_far, scaler_y_far = load_lstm_and_scalers("Far")
    except Exception as e:
        print(f"Ошибка загрузки LSTM моделей: {str(e)}")
        return

    preds = []
    errors = []

    for i, (true_d, rssi) in enumerate(zip(true_distances, rssi_values)):
        print(f"\nТочка {i + 1}: RSSI = {rssi:.2f}, Истинное расстояние = {true_d:.2f} м")

        try:
            cluster = majority_vote(trees, rssi)

            if cluster == 0:
                pred = predict_distance_lstm(model_near, scaler_x_near, scaler_y_near, [rssi])
            else:
                pred = predict_distance_lstm(model_far, scaler_x_far, scaler_y_far, [rssi])

            error = (pred - true_d) ** 2
            preds.append(pred)
            errors.append(error)

            print(f"  Предсказанное расстояние: {pred:.2f} м")
            print(f"  Ошибка: {pred - true_d:.2f} м")
        except Exception as e:
            print(f"  Ошибка обработки: {str(e)}")
            continue

    if not errors:
        print("Не удалось обработать ни одной точки!")
        return

    mse = np.mean(errors)
    print(f"\nРезультаты:")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.4f} м²")
    print(f"Средняя абсолютная ошибка (MAE): {np.mean(np.abs(np.sqrt(errors))):.4f} м")
    print(f"Количество обработанных точек: {len(errors)}/{len(true_distances)}")

    # График ошибки
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(errors)), np.sqrt(errors), 'b.', label='Ошибка по точкам (RMSE)')
    plt.axhline(y=np.sqrt(mse), color='r', linestyle='-', label=f'Средняя ошибка (RMSE = {np.sqrt(mse):.2f} м)')
    plt.xlabel("Номер точки")
    plt.ylabel("Ошибка (м)")
    plt.title("Ошибка предсказания расстояния")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

def majority_vote(trees, rssi_point):
    votes = []
    for i, tree in enumerate(trees):
        pred = tree.predict([[rssi_point]])[0]
        votes.append(pred)
        print(f"  Дерево {i + 1}: кластер {pred}")
    final = max(set(votes), key=votes.count)
    print(f"  → Итоговое решение (голосование): кластер {final}")
    return final
if __name__ == "__main__":
    main()