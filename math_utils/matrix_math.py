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


# Вычисляем возможные минимальные и максимальные значения
def calculate_min_max(inverse_color_matrix):
    rgb_values = [0, 1]
    combinations = list(itertools.product(rgb_values, repeat=3))

    min_values = np.inf * np.ones(3)
    max_values = -np.inf * np.ones(3)

    for comb in combinations:
        rgb_vector = np.array(comb)
        transformed_values = np.matmul(inverse_color_matrix, rgb_vector)

        min_values = np.minimum(min_values, transformed_values)
        max_values = np.maximum(max_values, transformed_values)

    return min_values, max_values


# Получаем минимальные и максимальные значения
min_values, max_values = calculate_min_max(inverse_color_matrix)


# Функция для преобразования вектора и проверки его диапазона
def color_vector_transform(rgb_vector):
    # Преобразуем вектор с помощью обратной матрицы
    transformed_vector = np.matmul(inverse_color_matrix, rgb_vector)

    # Проверяем, выходят ли значения за пределы
    for i, val in enumerate(transformed_vector):
        if val < min_values[i] or val > max_values[i]:
            print(f"Ошибка! Значение канала {i} ({val}) выходит за пределы [{min_values[i]}, {max_values[i]}]")
      #  else:
      #      print(f"Значение канала {i} в пределах нормы: {val}")

    # Нормализуем вектор по каналам
    normalized_vector = (transformed_vector - min_values) / (max_values - min_values)
    return normalized_vector


# Main для тестирования
if __name__ == "__main__":
    # Тестируем на векторе (1, 0.5, 0.3)
    example_vector = np.array([1, 0.5, 0.3])
    print(f"Тестовый вектор: {example_vector}")

    normalized_vector = transform_and_check(example_vector)

    print(f"Нормализованный вектор: {normalized_vector}")
