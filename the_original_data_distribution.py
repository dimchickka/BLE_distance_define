import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("data.txt", header=None, names=["Distance", "RSSI"])

# Сортировка по расстоянию для красоты
df = df.sort_values(by="Distance")

# График: просто все точки
plt.figure(figsize=(14, 6))
plt.scatter(df["Distance"], df["RSSI"], alpha=0.4, s=10, label="Измерения")

# Оформление
plt.title("Зависимость RSSI от расстояния (исходные данные)")
plt.xlabel("Расстояние (м)")
plt.ylabel("RSSI (dBm)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
