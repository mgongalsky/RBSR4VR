# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import random
import numpy as np
import cv2
import pickle
import tqdm

from torchvision.utils import save_image
from models.loss.image_quality_v2 import PSNR, SSIM, LPIPS
from models.alignment.pwcnet import PWCNet
from dataset.burstsr_dataset import QoocamPNGImage
from dataset.burstsr_dataset import BurstSRDataset
from verbosing_config import verb_data_dim_analysis  # Импортируем уровень вербозинга


env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_qoocam_val_set(root=None, split='val_mine'):
    """ Get the Qoocam validation dataset """
    burstsr_dataset = BurstSRDataset(split=split, root=root, initialize=True)
    burst_size = 14

    # Использование QoocamPNGImage.load
    dataset = [
        {
            'burst': [QoocamPNGImage.load(os.path.join(root, burst_name, f'qoocam_{i:02d}', 'im_raw.png')) for i in range(burst_size)],
            #'frame_gt': CanonImage.load(os.path.join(root, burst_name, 'canon')).get_image_data(),
            'burst_name': burst_name
        }
        for burst_name in burstsr_dataset.burst_list
    ]

    return dataset




def compute_score_BIPnet(model, model_path, num_frame, dataset_root='my_dataset/burstsr_dataset/val_mine'):
    from utils.metrics import AlignedPSNR, AlignedLPIPS, AlignedSSIM
    device = 'cuda'

    checkpoint_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['net'])
    model = model.to(device).train(False)
    model.eval()

    alignment_net = PWCNet(load_pretrained=True, weights_path='pretrained_networks/pwcnet-network-default.pth')
    alignment_net = alignment_net.cuda()
    alignment_net = alignment_net.eval()

    aligned_psnr_fn = AlignedPSNR(alignment_net=alignment_net)
    aligned_ssim_fn = AlignedSSIM(alignment_net=alignment_net)

    dataset = get_qoocam_val_set(root=dataset_root)
    PSNR = []
    SSIM = []

    import matplotlib.pyplot as plt

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        # gt = data['frame_gt'].get_image_data().unsqueeze(0)
        burst = torch.stack([frame.get_image_data() for frame in data['burst']]).unsqueeze(0)
        burst_name = data['burst_name']

        burst = burst.to(device)
        # gt = gt.to(device)

        # Вербозинг: вывод RGB изображения для каждого кадра в burst
        for frame_idx in range(burst.shape[1]):
            frame_tensor = burst[0, frame_idx]  # Извлекаем кадр из burst
            frame_np = frame_tensor.permute(1, 2, 0).cpu().numpy()  # Транспонируем из (C, H, W) в (H, W, C)

            # Учитываем, что у нас 4 канала, мы хотим сделать RGB изображение, игнорируя четвертый канал
            frame_rgb = frame_np[:, :, :3]

            # Масштабируем значения от [0, 1] до [0, 255] для корректного отображения
            frame_rgb = (frame_rgb * 255).astype(np.uint8)

            # Отображаем картинку
            #plt.imshow(frame_rgb)
            #plt.title(f"Frame {frame_idx} from Burst {idx}")
            #plt.axis('off')
            #plt.show()

        import torch.nn.functional as F  # Не забудь добавить этот импорт

        with torch.no_grad():
            if verb_data_dim_analysis >= 1:
                print(f"[V1:verb_data_dim_analysis] torch.no_grad() enabled for inference.")

            burst = burst[:, :num_frame, ...]
            net_pred, _ = model(burst)

            # Вербозинг: вывод максимальных и минимальных значений для каждого канала до масштабирования
            if verb_data_dim_analysis >= 2:
                max_values = net_pred.max(dim=2)[0].max(dim=2)[0]  # Максимумы по H и W
                min_values = net_pred.min(dim=2)[0].min(dim=2)[0]  # Минимумы по H и W
                max_values_str = ', '.join([f'C{c}: {max_values[0, c].item():.4f}' for c in range(max_values.shape[1])])
                min_values_str = ', '.join([f'C{c}: {min_values[0, c].item():.4f}' for c in range(min_values.shape[1])])
                print(f"[V2:verb_data_dim_analysis] Max values per channel before scaling - {max_values_str}")
                print(f"[V2:verb_data_dim_analysis] Min values per channel before scaling - {min_values_str}")

            # Масштабирование значений к диапазону [0, 1]
            output = net_pred #/ 1024.0  # Приводим значения в диапазон [0, 1]

            # Verbosing после масштабирования
            if verb_data_dim_analysis >= 2:
                max_values_scaled = output.max(dim=2)[0].max(dim=2)[0]
                min_values_scaled = output.min(dim=2)[0].min(dim=2)[0]
                max_values_scaled_str = ', '.join(
                    [f'C{c}: {max_values_scaled[0, c].item():.4f}' for c in range(max_values_scaled.shape[1])])
                min_values_scaled_str = ', '.join(
                    [f'C{c}: {min_values_scaled[0, c].item():.4f}' for c in range(min_values_scaled.shape[1])])
                print(f"[V2:verb_data_dim_analysis] Max values per channel after scaling - {max_values_scaled_str}")
                print(f"[V2:verb_data_dim_analysis] Min values per channel after scaling - {min_values_scaled_str}")

            # Обрезка значений тензора (чтобы все значения были точно в диапазоне [0, 1])
            output = output.clamp(0.0, 1.0)

            # Сохранение основного изображения
            save_path = os.path.join(output_dir, f'output_{idx}.png')
            save_image(output.squeeze(), save_path)
            print(f"[V1:verb_data_dim_analysis] Image saved at {save_path}")

            # Вербозинг: размерности перед интерполяцией
            if verb_data_dim_analysis >= 2:
                print(f"[V2:verb_data_dim_analysis] Output shape before interpolation: {output.shape}")

            # Приведение тензора к нужному формату (N, C, H, W) — предполагаем, что output уже в формате (N, C, H, W)
            # Уменьшение изображения до 80x80 с билинейной интерполяцией
            output_resized = F.interpolate(output, size=(80, 80), mode='bilinear', align_corners=False)

            if verb_data_dim_analysis >= 2:
                print(f"[V2:verb_data_dim_analysis] Resized output shape: {output_resized.shape}")

            # Сохранение уменьшенного изображения
            resized_save_path = os.path.join(output_dir, f'output_resized_{idx}.png')
            save_image(output_resized.squeeze(), resized_save_path)
            print(f"[V1:verb_data_dim_analysis] Resized image saved at {resized_save_path}")


if __name__ == "__main__":
    import csv

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    setup_seed(0)
    from models.RBSR_test import RBSR

    net = RBSR()
    path = "./pretrained_networks/RBSR_realwporld.pth.tar"
    #psnr, ssim = (
    compute_score_BIPnet(net, path, 14)
