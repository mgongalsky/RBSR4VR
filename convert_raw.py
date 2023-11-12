import cv2
import numpy as np
import torch
import pickle as pkl
from torchvision.transforms.functional import to_pil_image
from dataset.burstsr_dataset import CanonImage


# Замените 'path_to_your_data' на путь к вашим данным
path_to_your_data = './example_images'

# Загрузка изображения и метаданных
im_raw = cv2.imread(f'{path_to_your_data}/im_raw_canon.png', cv2.IMREAD_UNCHANGED)
im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.float32)
im_raw = torch.from_numpy(im_raw)
meta_data = pkl.load(open(f'{path_to_your_data}/meta_info_canon.pkl', "rb"))

# Создание экземпляра CanonImage
canon_image = CanonImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                         meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])

# Преобразование RAW в RGB
processed_image = CanonImage.generate_processed_image(canon_image.get_image_data(substract_black_level=True, 
                                                                                 white_balance=False, 
                                                                                 normalize=False),
                                                      meta_data, return_np=True)

# Сохранение изображения
output_image = to_pil_image(processed_image)
output_image.save(f'{path_to_your_data}/output_canon_image.png')
