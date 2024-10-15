import numpy as np
import itertools

# Исходная матрица преобразования
color_matrix = np.array([
    [1.2429526, -0.15225857, -0.09069402],
    [-0.18403465, 1.4261994, -0.24216475],
    [-0.02236049, -0.6038957, 1.6262562]
])

# Вычисляем обратную матрицу
inverse_color_matrix = np.linalg.pinv(color_matrix)

# Возможные значения для R, G и B (0 или 1)
rgb_values = [0, 1]

# Все возможные комбинации RGB (0, 1)
combinations = list(itertools.product(rgb_values, repeat=3))

# Расчет минимальных и максимальных значений для каждого канала
min_values = np.inf * np.ones(3)
max_values = -np.inf * np.ones(3)

for comb in combinations:
    rgb_vector = np.array(comb)
    transformed_values = np.matmul(inverse_color_matrix, rgb_vector)

    min_values = np.minimum(min_values, transformed_values)
    max_values = np.maximum(max_values, transformed_values)

print(f"Минимальные значения в каждом канале: {min_values}")
print(f"Максимальные значения в каждом канале: {max_values}")

# Теперь нормализуем примерный вектор
example_vector = np.array([0, 1, 1])

# Преобразуем вектор с помощью обратной матрицы
transformed_vector = np.matmul(inverse_color_matrix, example_vector)

# Проверка, выходят ли значения за пределы
for i, val in enumerate(transformed_vector):
    if val < min_values[i] or val > max_values[i]:
        print(f"Ошибка! Значение канала {i} ({val}) выходит за пределы [{min_values[i]}, {max_values[i]}]")

# Нормализуем его по каналам
normalized_vector = (transformed_vector - min_values) / (max_values - min_values)

print(f"Нормализованный вектор: {normalized_vector}")
