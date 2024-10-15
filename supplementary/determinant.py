import numpy as np

# Определение матрицы
color_matrix = np.array([
    [1.2429526, -0.15225857, -0.09069402],
    [-0.18403465, 1.4261994, -0.24216475],
    [-0.02236049, -0.6038957, 1.6262562]
])

# Вычисление детерминанта
determinant = np.linalg.det(color_matrix)
print(determinant)

import numpy as np

# Определение матрицы
color_matrix = np.array([
    [1.2429526, -0.15225857, -0.09069402],
    [-0.18403465, 1.4261994, -0.24216475],
    [-0.02236049, -0.6038957, 1.6262562]
])

# Определение вектора
vector = np.array([1, 1, 1])

# Умножение матрицы на вектор
result = np.dot(color_matrix, vector)

# Вывод результата
print(result)

