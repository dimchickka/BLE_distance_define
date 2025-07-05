import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# === Загрузка данных ===
data = np.loadtxt("LearningData.txt", delimiter=",")
distances = data[:, 0].reshape(-1, 1)
rssi = data[:, 1].reshape(-1, 1)

X = np.hstack([rssi, distances])  # Включаем расстояние и RSSI в признаки

# === Построение AutoEncoder ===
input_dim = X.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoded = Dense(4, activation='relu')(input_layer)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(4, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
encoder = Model(inputs=input_layer, outputs=encoded)

autoencoder.compile(optimizer=Adam(0.001), loss='mse')

# === Обучение AutoEncoder ===
autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, verbose=0)

# === Кодирование признаков и кластеризация ===
encoded_X = encoder.predict(X)
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(encoded_X)

# === Сохранение в два файла ===
cluster_0 = data[cluster_labels == 0]
cluster_1 = data[cluster_labels == 1]

np.savetxt("Cluster0.txt", cluster_0, fmt="%.2f", delimiter=",")
np.savetxt("Cluster1.txt", cluster_1, fmt="%.2f", delimiter=",")

# === Вывод статистики ===
for label, cluster_data in zip([0, 1], [cluster_0, cluster_1]):
    print(f"Кластер {label}: {len(cluster_data)} точек, диапазон расстояний {cluster_data[:,0].min():.2f} - {cluster_data[:,0].max():.2f} м")
