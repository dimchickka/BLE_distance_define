import numpy as np

# Загрузка исходных данных
data = np.loadtxt('LearningData.txt', delimiter=',')

# Разделение на две категории
near_mask = (data[:, 0] >= 0.1) & (data[:, 0] <= 2.2)
far_mask = (data[:, 0] > 2.2) & (data[:, 0] <= 5.0)

near_data = data[near_mask]
far_data = data[far_mask]

# Сохранение данных в соответствующие файлы
np.savetxt('LearningDataForLSTMNear.txt', near_data, delimiter=',', fmt='%.2f')
np.savetxt('LearningDataForLSTMFar.txt', far_data, delimiter=',', fmt='%.2f')

print("Файлы успешно сохранены:")
print(f"  - LearningDataForLSTMNear.txt: {len(near_data)} строк")
print(f"  - LearningDataForLSTMFar.txt: {len(far_data)} строк")
