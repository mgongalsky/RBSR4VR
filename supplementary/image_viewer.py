from PIL import Image
import matplotlib.pyplot as plt

# Открытие PNG файла
def open_and_show_image_no_interpolation(image_path, scale_factor=5):
    # Открываем изображение с помощью Pillow
    img = Image.open(image_path)

    # Задаем размер окна и выводим изображение с отключенной интерполяцией
    plt.figure(figsize=(img.size[0] / 100 * scale_factor, img.size[1] / 100 * scale_factor))

    # Отображаем изображение с отключенной интерполяцией
    plt.imshow(img, interpolation='nearest')

    # Убираем оси для чистоты изображения
    plt.axis('off')

    # Показать изображение
    plt.show()

# Пример использования
image_path = '../output/output_3.png'
open_and_show_image_no_interpolation(image_path)
