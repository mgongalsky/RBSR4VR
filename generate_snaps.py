# import NFOV from eq2rect.py
from eq2rect import NFOV
import imageio.v2 as imageio
import numpy as np
import os
import cv2

def generate_snapshots_from_video(video_path, output_folder, frame_numbers, img_size=(128, 128), overlap_factor=0.8):
    # Открытие видеофайла
    video = cv2.VideoCapture(video_path)
    current_frame = 0
    success, frame = video.read()

    while success:
        if current_frame in frame_numbers:
            print(f"Processing {current_frame} frame now:", end='\r')

            # Конвертация изображения из BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_folder = os.path.join(output_folder, str(current_frame))
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

            # Обработка кадра
            frame_height, frame_width = frame.shape[:2]
            generate_snapshots(frame, frame_folder, frame_size=(frame_width, frame_height), img_size=img_size, overlap_factor=overlap_factor)


        current_frame += 1
        success, frame = video.read()

    video.release()
def generate_snapshots(input_data, output_folder, frame_size=None, img_size=(128, 128), overlap_factor=0.8):
    # Проверяем, передается ли путь к файлу или numpy массив
    if isinstance(input_data, str):  # Если это путь к файлу
        img = imageio.imread(input_data, pilmode='RGB')
        frame_height, frame_width = img.shape[:2]
    elif isinstance(input_data, np.ndarray):  # Если это numpy массив (кадр из видео)
        img = input_data
        if frame_size is None:
            frame_height, frame_width = img.shape[:2]
        else:
            frame_width, frame_height = frame_size
    else:
        raise ValueError("Input data must be a file path or numpy array")

    # Создание директории для сохранения, если она еще не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Загрузка исходного сферического изображения
    #import imageio.v2 as imageio
    #img = imageio.imread(input_image_path, pilmode='RGB')

    # Определение FOV
    h_fov = (img_size[0] / frame_size[0]) * np.pi
    v_fov = (img_size[1] / frame_size[1]) * np.pi
    nfov = NFOV(height=img_size[1], width=img_size[0], h_fov=h_fov, v_fov=v_fov)
    #nfov.FOV = [h_fov, v_fov]

    # Определение шагов для перебора координат с учетом перекрытия
    lat_steps = int((frame_size[1] / img_size[1]) * overlap_factor)
    lon_steps = int((frame_size[0] / img_size[0]) * overlap_factor)


    #total_images = (frame_size[1] // img_size[1]) * (frame_size[0] // img_size[0])
    total_images = lat_steps * lon_steps

    images_processed = 0

    for lat in np.linspace(0, 1, lat_steps, endpoint=False):
        for lon in np.linspace(0, 1, lon_steps, endpoint=False):
            center_point = np.array([lon, lat])
            snapshot = nfov.toNFOV(img, center_point)

            # Форматирование значений координат
            formatted_lon = f"{lon:.4f}"
            formatted_lat = f"{lat:.4f}"
            filename = os.path.join(output_folder, f"snapshot_{formatted_lon}_{formatted_lat}.png")
            imageio.imwrite(filename, snapshot)

            # Обновление прогресса
            images_processed += 1
            progress = (images_processed / total_images) * 100
            print(f"Processed {images_processed}/{total_images} images ({progress:.2f}%)", end='\r')

    print("\nAll images have been processed.")

# Загрузка исходного сферического изображения для определения его размеров
#input_image_path = 'vr_equi_test.png'
#img = imageio.imread(input_image_path, pilmode='RGB')
#frame_height, frame_width = img.shape[:2]

# Вызов функции с размерами исходного изображения
#generate_snapshots(input_image_path, 'VR_snapshots', frame_size=(frame_width, frame_height))

# Пример вызова функции
video_path = 'VR_test_4K.MP4'
generate_snapshots_from_video(video_path, 'VR_snapshots', [10, 20])

