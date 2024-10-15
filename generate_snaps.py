# import NFOV from eq2rect.py
from eq2rect import NFOV
import imageio.v2 as imageio
import numpy as np
import os
import cv2
import pickle
import random
import shutil


def generate_snapshots_from_video(video_path, output_folder, start_frame=20, num_frames=14, num_series=10, img_size=(80, 80), overlap_factor=0.8):
    # Открытие видеофайла
    video = cv2.VideoCapture(video_path)
    current_frame = 0
    success, frame = video.read()

    # Обработка кадров, начиная с `start_frame` и до `start_frame + num_frames`
    end_frame = start_frame + num_frames

    all_coords = []  # Для сохранения всех возможных координат патчей
    snapshot_map = {}  # Для хранения патчей и их соответствующих координат

    # Считывание всех кадров и сохранение патчей в snapshot_map
    while success and current_frame < end_frame:
        if current_frame >= start_frame:
            print(f"Processing frame {current_frame} now...", end='\r')

            # Конвертация изображения из BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Получение всех патчей и координат
            patches, coords = generate_snapshots(frame, frame_size=(frame.shape[1], frame.shape[0]), img_size=img_size, overlap_factor=overlap_factor)
            snapshot_map[current_frame - start_frame] = patches
            all_coords.extend(coords)

        current_frame += 1
        success, frame = video.read()

    video.release()

    # Выбор случайных координат для серий
    random.seed(42)
    chosen_coords = random.sample(all_coords, num_series)

    # Создание структуры папок для серий
    for series_idx, coord in enumerate(chosen_coords):
        # Основная папка серии, теперь просто нумеруем (0000, 0001, ... 0009)
        series_folder = os.path.join(output_folder, f'{series_idx:04d}')
        os.makedirs(series_folder, exist_ok=True)

        # Создание папки canon и копирование файлов
        canon_folder = os.path.join(series_folder, 'canon')
        os.makedirs(canon_folder, exist_ok=True)
        shutil.copy('im_raw.png', canon_folder)
        shutil.copy('meta_info.pkl', canon_folder)

        # Создание папок для 14 кадров (qoocam_00 - qoocam_13)
        for frame_idx in range(14):
            frame_folder = os.path.join(series_folder, f'qoocam_{frame_idx:02d}')
            os.makedirs(frame_folder, exist_ok=True)

            # Сохранение соответствующего патча
            patch = snapshot_map[frame_idx][coord]
            filename = os.path.join(frame_folder, 'im_raw.png')  # Переименовали в im_raw.png
            imageio.imwrite(filename, patch)

    print("\nAll images have been processed and saved.")


def generate_snapshots(input_data, frame_size=None, img_size=(80, 80), overlap_factor=0.8):
    # Проверяем, передается ли путь к файлу или numpy массив
    if isinstance(input_data, str):  # Если это путь к файлу
        img = imageio.imread(input_data, pilmode='RGB')
    elif isinstance(input_data, np.ndarray):  # Если это numpy массив (кадр из видео)
        img = input_data
    else:
        raise ValueError("Input data must be a file path or numpy array")

    frame_height, frame_width = frame_size
    h_fov = (img_size[0] / frame_width) * np.pi
    v_fov = (img_size[1] / frame_height) * np.pi
    nfov = NFOV(height=img_size[1], width=img_size[0], h_fov=h_fov, v_fov=v_fov)

    lat_steps = int((frame_height / img_size[1]) * overlap_factor)
    lon_steps = int((frame_width / img_size[0]) * overlap_factor)

    patches = {}
    coords = []

    for lat in np.linspace(0, 1, lat_steps, endpoint=False):
        for lon in np.linspace(0, 1, lon_steps, endpoint=False):
            center_point = np.array([lon, lat])
            snapshot = nfov.toNFOV(img, center_point)
            coords.append((lon, lat))
            patches[(lon, lat)] = snapshot

    return patches, coords

# Пример вызова функции
video_path = 'VR_test_4K.MP4'
generate_snapshots_from_video(video_path, 'VR_snapshots', start_frame=20, num_frames=14, num_series=10, img_size=(80, 80))
