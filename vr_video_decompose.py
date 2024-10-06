import cv2
import os
import numpy as np
from py360convert import e2r

# Настройки для обработки видео
video_path = 'VR_test_4K.MP4'  # Укажите путь к файлу видео
frame_ids = [10, 20]  # Кадры, которые будут обработаны
output_folder = 'VR_snapshots'  # Папка для сохранения результатов
img_size = (3456, 1728)  # Размер исходного видео
target_size = (128, 128)  # Размер целевого изображения
fov = 90  # Поле зрения для rectilinear изображения
theta = np.arange(-180, 180, 45)  # Диапазон широты в градусах
phi = np.arange(-90, 90, 45)  # Диапазон долготы в градусах

# Создаем папку для сохранения результатов
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Загружаем видео
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обрабатываем только выбранные кадры
    if frame_count in frame_ids:
        frame_folder = os.path.join(output_folder, f'frame_{frame_count}')
        os.makedirs(frame_folder, exist_ok=True)

        # Переводим equirectangular в rectilinear
        for t in theta:
            for p in phi:
                rect_img = e2r(frame, target_size, fov, [t, p], mode='bilinear')
                filename = f'snapshot_theta{t}_phi{p}.jpg'
                cv2.imwrite(os.path.join(frame_folder, filename), rect_img)

    frame_count += 1

cap.release()
