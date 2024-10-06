import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

input_image_path = 'test_raw.png'

def process_image(path):
    # Загрузка изображения
    im_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im_raw is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Вербозинг: размерность после загрузки
    print(f"Initial shape (BGR): {im_raw.shape}")

    # Перевод из BGR в RGB
    im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB)
    print(f"Shape after converting BGR to RGB: {im_raw.shape}")

    # Нормализация в диапазон [0, 1]
    im_raw = im_raw.astype(np.float32) / 255.0
    print(f"Shape after normalization: {im_raw.shape}")

    # Определение исходной матрицы (RGB -> Samsung RAW)
    original_color_matrix = np.array([
        [1.2429526, -0.15225857, -0.09069402, 0.0],
        [-0.18403465, 1.4261994, -0.24216475, 0.0],
        [-0.02236049, -0.6038957, 1.6262562, 0.0]
    ])

    # Определение исходной матрицы (RGB -> Samsung RAW)
    #original_color_matrix = np.array([
    #    [1, 0, 0, 0.0],
    #    [0, 1, 0, 0.0],
    #    [0, 0, 1, 0.0]
    #])

    # Вычисление псевдообратной матрицы
    color_matrix_pinv = np.linalg.pinv(original_color_matrix[:, :3])
    print(f"Inverse color matrix shape: {color_matrix_pinv.shape}")

    # Преобразование изображения через матрицу
    height, width, _ = im_raw.shape
    reshaped_image = im_raw.reshape(-1, 3)
    print(f"Shape before matrix multiplication (reshaped): {reshaped_image.shape}")
    transformed_image = np.matmul(reshaped_image, color_matrix_pinv.T)
    print(f"Shape after matrix multiplication: {transformed_image.shape}")
    transformed_image = np.clip(transformed_image, 0, 1)

    # Возвращаем изображение в оригинальную форму
    transformed_image = transformed_image.reshape(height, width, 3)
    print(f"Shape after reshaping back: {transformed_image.shape}")

    # Добавление 4-го канала (дублирование зеленого канала)
    green_channel = transformed_image[:, :, 1]
    im_raw = np.dstack((transformed_image[:, :, 0], green_channel, green_channel, transformed_image[:, :, 2]))
    print(f"Shape after adding 4th channel: {im_raw.shape}")

    # Масштабируем изображение для совместимости с Samsung RAW
    im_raw = im_raw * 1023.0
    im_raw = im_raw.astype(np.int16)
    print(f"Shape after scaling and type conversion: {im_raw.shape}")

    # Преобразуем изображение в тензор PyTorch
    im_tensor = torch.from_numpy(im_raw)
    print(f"Tensor shape: {im_tensor.shape}")

    return im_tensor

def display_image(image_tensor, title="Processed Image"):
    # Вербозинг для проверки формы перед отображением
    print(f"Tensor shape before displaying: {image_tensor.shape}")

    # Исправим: image_tensor должен быть (C, H, W), а нам нужно преобразовать в (H, W, C)
    # Используем .permute(1, 2, 0) чтобы получить корректную форму для отображения
    image_np = image_tensor.cpu().numpy()  # (H, W, C)
    print(f"Shape after permute: {image_np.shape}")

    # Используем только первые три канала (RGB) и масштабируем в диапазон [0, 255]
    image_rgb = image_np[:, :, :3]
    image_rgb = np.clip(image_rgb / 1023.0 * 255.0, 0, 255).astype(np.uint8)

    # Отображение
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()



# Основная функция для обработки и отображения изображения
def main():
    try:
        processed_tensor = process_image(input_image_path)
        display_image(processed_tensor, title="Processed Image from test_raw.png")
    except FileNotFoundError as e:
        print(str(e))

if __name__ == "__main__":
    main()
