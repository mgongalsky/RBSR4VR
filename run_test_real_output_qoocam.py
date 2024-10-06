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

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
#        gt = data['frame_gt'].get_image_data().unsqueeze(0)
        burst = torch.stack([frame.get_image_data() for frame in data['burst']]).unsqueeze(0)
        burst_name = data['burst_name']

        burst = burst.to(device)
       # gt = gt.to(device)

        with torch.no_grad():
            burst = burst[:, :num_frame, ...]
            net_pred, _ = model(burst)
            output = net_pred.clamp(0.0, 1.0)
            save_path = os.path.join(output_dir, f'output_{idx}.png')
            save_image(output.squeeze(), save_path)

       # PSNR_temp = aligned_psnr_fn(output, gt, burst).cpu().numpy()
       # PSNR.append(PSNR_temp)

       # SSIM_temp = aligned_ssim_fn(output, gt, burst).cpu().numpy()
       # SSIM.append(SSIM_temp)

    #print(f'PSNR: {np.mean(np.array(PSNR))}, SSIM: {np.mean(np.array(SSIM))}')
    #return np.mean(np.array(PSNR)), np.mean(np.array(SSIM))


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
