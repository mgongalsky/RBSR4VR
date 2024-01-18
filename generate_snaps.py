# import NFOV from eq2rect.py
from eq2rect import NFOV
import numpy as np
import os

def generate_snapshots(input_image_path, output_folder, frame_size=(3840, 1920), img_size=(128, 128), overlap_factor=0.8):
    # Создание директории для сохранения, если она еще не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Загрузка исходного сферического изображения
    import imageio.v2 as imageio
    img = imageio.imread(input_image_path, pilmode='RGB')

    # Определение FOV
    h_fov = (img_size[0] / frame_size[0]) * np.pi
    v_fov = (img_size[1] / frame_size[1]) * np.pi
    nfov = NFOV(height=img_size[1], width=img_size[0])
    nfov.FOV = [h_fov, v_fov]

    # Определение шагов для перебора координат с учетом перекрытия
    lat_steps = int((frame_size[1] / img_size[1]) * overlap_factor)
    lon_steps = int((frame_size[0] / img_size[0]) * overlap_factor)


    #total_images = (frame_size[1] // img_size[1]) * (frame_size[0] // img_size[0])
    total_images = lat_steps * lon_steps

    images_processed = 0

    # Генерация изображений для различных сферических координат
    #for lat in np.linspace(0, 1, frame_size[1] // img_size[1]):
    #    for lon in np.linspace(0, 1, frame_size[0] // img_size[0]):
    #        center_point = np.array([lon, lat])
    #        snapshot = nfov.toNFOV(img, center_point)
    #        filename = os.path.join(output_folder, f"snapshot_{lon}_{lat}.png")
    #        imageio.imwrite(filename, snapshot)

    for lat in np.linspace(0, 1, lat_steps, endpoint=False):
        for lon in np.linspace(0, 1, lon_steps, endpoint=False):
    #for lat in np.linspace(0, 1, frame_size[1] // img_size[1], endpoint=False):
     #   for lon in np.linspace(0, 1, frame_size[0] // img_size[0], endpoint=False):
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

# Пример использования
generate_snapshots('vr_equi_test.png', 'VR_snapshots')
