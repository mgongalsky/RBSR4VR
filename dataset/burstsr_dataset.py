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
import torch
import cv2
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import zipfile
import shutil
from dataset.base_rawburst_dataset import BaseRawBurstDataset
from admin.environment import env_settings
from data import processing, sampler
from verbosing_config import verb_data_dim_analysis  # Импортируем уровень вербозинга


def load_txt(path):
    with open(path, 'r') as fh:
        out = [d.rstrip() for d in fh.readlines()]

    return out


class SamsungRAWImage:
    """ Custom class for RAW images captured from Samsung Galaxy S8 """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return SamsungRAWImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                               meta_data['daylight_wb'], meta_data['color_matrix'], meta_data['exif_data'],
                               meta_data.get('im_preview', None))

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, color_matrix, exif_data, im_preview=None):
        self.im_raw = im_raw
        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.color_matrix = color_matrix
        self.exif_data = exif_data
        self.im_preview = im_preview

        self.norm_factor = 1023.0

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'color_matrix': self.color_matrix.tolist()}

    def get_exposure_time(self):
        return self.exif_data['Image ExposureTime'].values[0].decimal()

    def get_noise_profile(self):
        noise = self.exif_data['Image Tag 0xC761'].values
        noise = [n[0] for n in noise]
        noise = np.array(noise).reshape(3, 2)
        return noise

    def get_f_number(self):
        return self.exif_data['Image FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['Image ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]

        if self.im_preview is not None:
            im_preview = self.im_preview[2*r1:2*r2, 2*c1:2*c2]
        else:
            im_preview = None

        return SamsungRAWImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.color_matrix,
                               self.exif_data, im_preview=im_preview)

    def postprocess(self, return_np=True, norm_factor=None):
        raise NotImplementedError


class CanonImage:
    """ Custom class for RAW images captured from Canon DSLR """
    @staticmethod
    def load(path):
        im_raw = cv2.imread('{}/im_raw.png'.format(path), cv2.IMREAD_UNCHANGED)
        im_raw = np.transpose(im_raw, (2, 0, 1)).astype(np.int16)
        im_raw = torch.from_numpy(im_raw)
        meta_data = pkl.load(open('{}/meta_info.pkl'.format(path), "rb", -1))

        return CanonImage(im_raw.float(), meta_data['black_level'], meta_data['cam_wb'],
                          meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])

    @staticmethod
    def generate_processed_image(im, meta_data, return_np=False, external_norm_factor=None, gamma=True, smoothstep=True,
                                 no_white_balance=False):
        im = im * meta_data.get('norm_factor', 1.0)

        if not meta_data.get('black_level_subtracted', False):
            im = (im - torch.tensor(meta_data['black_level'])[[0, 1, -1]].view(3, 1, 1))

        if not meta_data.get('while_balance_applied', False) and not no_white_balance:
            im = im * torch.tensor(meta_data['cam_wb'])[[0, 1, -1]].view(3, 1, 1) / torch.tensor(meta_data['cam_wb'])[1]

        im_out = im

        if external_norm_factor is None:
            im_out = im_out / (im_out.mean() * 5.0)
        else:
            im_out = im_out / external_norm_factor

        im_out = im_out.clamp(0.0, 1.0)

        if gamma:
            im_out = im_out ** (1.0 / 2.2)

        if smoothstep:
            # Smooth curve
            im_out = 3 * im_out ** 2 - 2 * im_out ** 3

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() * 255.0
            im_out = im_out.astype(np.uint8)
        return im_out

    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data):
        super(CanonImage, self).__init__()
        self.im_raw = im_raw

        if len(black_level) == 4:
            black_level = [black_level[0], black_level[1], black_level[3]]
        self.black_level = black_level

        if len(cam_wb) == 4:
            cam_wb = [cam_wb[0], cam_wb[1], cam_wb[3]]
        self.cam_wb = cam_wb

        if len(daylight_wb) == 4:
            daylight_wb = [daylight_wb[0], daylight_wb[1], daylight_wb[3]]
        self.daylight_wb = daylight_wb

        self.rgb_xyz_matrix = rgb_xyz_matrix

        self.exif_data = exif_data

        self.norm_factor = 16383

    def shape(self):
        shape = (3, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def get_all_meta_data(self):
        return {'black_level': self.black_level, 'cam_wb': self.cam_wb, 'daylight_wb': self.daylight_wb,
                'rgb_xyz_matrix': self.rgb_xyz_matrix.tolist(), 'norm_factor': self.norm_factor}

    def get_exposure_time(self):
        return self.exif_data['EXIF ExposureTime'].values[0].decimal()

    def get_f_number(self):
        return self.exif_data['EXIF FNumber'].values[0].decimal()

    def get_iso(self):
        return self.exif_data['EXIF ISOSpeedRatings'].values[0]

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(3, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(3, 1, 1) / 1024.0

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def set_image_data(self, im_data):
        self.im_raw = im_data

    def crop_image(self, r1, r2, c1, c2):
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return CanonImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
                          self.exif_data)

    def set_crop_info(self, crop_info):
        self.crop_info = crop_info

    def resize(self, size=None, scale_factor=None):
        self.im_raw = F.interpolate(self.im_raw.unsqueeze(0), size=size, scale_factor=scale_factor,
                                    mode='bilinear').squeeze(0)

    def postprocess(self, return_np=True):
        raise NotImplementedError


import os
import torch
import cv2
import numpy as np

import numpy as np
import torch
import cv2

import cv2
import numpy as np
import torch
from verbosing_config import verb_data_dim_analysis  # Импортируем уровень вербозинга

class QoocamPNGImage:
    """ Custom class for handling images captured from Qoocam in PNG format """

    @staticmethod
    def load(path):
        """
        Load an image from the specified path.
        Args:
            path: Path to the PNG image.
        Returns:
            QoocamPNGImage object.
        """
        # Load image as RGB
        im_raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if im_raw is None:
            raise FileNotFoundError(f"Image not found at path: {path}")

        # Print the initial dimensions of im_raw (BGR) if verbosing level is 1 or higher
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] Initial im_raw shape (BGR): {im_raw.shape}")

        # Convert BGR (used by OpenCV) to RGB
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB)
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] im_raw shape after BGR to RGB conversion: {im_raw.shape}")

        # Normalize to the range [0, 1] for matrix multiplication
        im_raw = im_raw.astype(np.float32) / 255.0
        if verb_data_dim_analysis >= 2:  # Detализация на уровне 2
            print(f"[V2:verb_data_dim_analysis] im_raw shape after normalization to [0, 1]: {im_raw.shape}")

        # Define the RGB to Samsung transformation matrix and calculate its inverse
        color_matrix = np.array([[1.2429526, -0.15225857, -0.09069402, 0.0],
                                 [-0.18403465, 1.4261994, -0.24216475, 0.0],
                                 [-0.02236049, -0.6038957, 1.6262562, 0.0]])
        inverse_color_matrix = np.linalg.pinv(color_matrix[:, :3])

        # Perform matrix multiplication to transform RGB to Samsung-RAW-like format
        height, width, _ = im_raw.shape
        reshaped_image = im_raw.reshape(-1, 3)
        if verb_data_dim_analysis >= 2:
            print(f"[V2:verb_data_dim_analysis] reshaped_image shape before matrix multiplication: {reshaped_image.shape}")

        transformed_image = np.matmul(reshaped_image, inverse_color_matrix.T)
        transformed_image = np.clip(transformed_image, 0, 1)
        if verb_data_dim_analysis >= 2:
            print(f"[V2:verb_data_dim_analysis] transformed_image shape after matrix multiplication: {transformed_image.shape}")

        # Reshape back to original image dimensions and convert to int16
        transformed_image = transformed_image.reshape(height, width, 3)
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] transformed_image shape after reshaping: {transformed_image.shape}")

        # Reorganize channels to have RGGB format (Red, Green, Green, Blue)
        red_channel = transformed_image[:, :, 0]
        green_channel = transformed_image[:, :, 1]
        blue_channel = transformed_image[:, :, 2]
        im_raw = np.dstack((red_channel, green_channel, green_channel, blue_channel))
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] im_raw shape after reorganizing channels to RGGB: {im_raw.shape}")

        # Scale the image to match expected input levels
        im_raw = im_raw * 1023.0
        im_raw = im_raw.astype(np.int16)
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] im_raw shape after scaling to int16: {im_raw.shape}")

        # Transpose to (C, H, W) format
        im_raw = np.transpose(im_raw, (2, 0, 1))
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] im_raw shape after transpose to (C, H, W): {im_raw.shape}")

        # Convert to PyTorch tensor
        im_raw = torch.from_numpy(im_raw)
        if verb_data_dim_analysis >= 1:
            print(f"[V1:verb_data_dim_analysis] Tensor shape: {im_raw.shape}")

        # Set some dummy metadata for compatibility
        meta_data = {
            'black_level': [0, 0, 0, 0],
            'cam_wb': [1.0, 1.0, 1.0, 1.0],
            'daylight_wb': [1.0, 1.0, 1.0, 1.0],
            'rgb_xyz_matrix': np.eye(3).tolist(),
            'exif_data': {}
        }

        return QoocamPNGImage(im_raw, meta_data['black_level'], meta_data['cam_wb'],
                              meta_data['daylight_wb'], meta_data['rgb_xyz_matrix'], meta_data['exif_data'])



    def __init__(self, im_raw, black_level, cam_wb, daylight_wb, rgb_xyz_matrix, exif_data):
        """
        Initialize the QoocamPNGImage object with the image data and metadata.
        """
        self.im_raw = im_raw
        self.black_level = black_level
        self.cam_wb = cam_wb
        self.daylight_wb = daylight_wb
        self.rgb_xyz_matrix = rgb_xyz_matrix
        self.exif_data = exif_data

        # Set normalization factor for the image
        self.norm_factor = 1023.0

    def get_image_data(self, substract_black_level=False, white_balance=False, normalize=False):
        """
        Apply image processing options to the loaded image.
        Args:
            substract_black_level: Whether to subtract black level from the image.
            white_balance: Whether to apply white balance to the image.
            normalize: Whether to normalize image values.
        Returns:
            Processed image tensor.
        """
        im_raw = self.im_raw.float()

        if substract_black_level:
            im_raw = im_raw - torch.tensor(self.black_level).view(4, 1, 1)

        if white_balance:
            im_raw = im_raw * torch.tensor(self.cam_wb).view(4, 1, 1)

        if normalize:
            im_raw = im_raw / self.norm_factor

        return im_raw

    def shape(self):
        """
        Get the shape of the image.
        Returns:
            Tuple representing the shape of the image.
        """
        shape = (4, self.im_raw.shape[1], self.im_raw.shape[2])
        return shape

    def crop_image(self, r1, r2, c1, c2):
        """
        Crop the image to the specified coordinates.
        Args:
            r1, r2: Row indices for cropping.
            c1, c2: Column indices for cropping.
        """
        self.im_raw = self.im_raw[:, r1:r2, c1:c2]

    def get_crop(self, r1, r2, c1, c2):
        """
        Get a cropped portion of the image.
        Args:
            r1, r2: Row indices for cropping.
            c1, c2: Column indices for cropping.
        Returns:
            Cropped QoocamPNGImage object.
        """
        im_raw = self.im_raw[:, r1:r2, c1:c2]
        return QoocamPNGImage(im_raw, self.black_level, self.cam_wb, self.daylight_wb, self.rgb_xyz_matrix,
                              self.exif_data)

    def postprocess(self, return_np=True):
        """
        Post-process the image (dummy method for now).
        Args:
            return_np: Whether to return the image as a NumPy array.
        Returns:
            Processed image.
        """
        im_out = self.im_raw.clamp(0, self.norm_factor)

        if return_np:
            im_out = im_out.permute(1, 2, 0).numpy() / self.norm_factor * 255.0
            im_out = im_out.astype(np.uint8)

        return im_out


class BurstSRDataset(BaseRawBurstDataset):
    """ Real-world burst super-resolution dataset. """
    def __init__(self, root=None, split='train', seq_ids=None, initialize=True):
        """
        args:
            root - Path to root dataset directory
            split - Can be 'train', 'val', or 'val_mine'
            seq_ids - (Optional) List of sequences to load. If specified, the 'split' argument is ignored.
            initialize - boolean indicating whether to load the meta-data for the dataset
        """
        root = env_settings().burstsr_dir if root is None else root
        super().__init__('BurstSRDataset', root)

        self.split = split
        self.seq_ids = seq_ids
        self.root = root

        if initialize:
            self.initialize()

        self.initialized = initialize

    def initialize(self):
        split = self.split
        seq_ids = self.seq_ids
        self.burst_list = self._get_burst_list(split, seq_ids)

    def _get_burst_list(self, split, seq_ids):
        # Проверяем, чтобы split не был дублированным
        if split == 'val_mine':
            burst_list_path = self.root #os.path.join(self.root, split)
        else:
            burst_list_path = os.path.join(self.root, self.split)

        print(f"[DEBUG] Looking for burst list in: {burst_list_path}")

        try:
            burst_list = sorted(os.listdir(burst_list_path))
            print(f"[DEBUG] Found burst list: {burst_list}")
        except FileNotFoundError as e:
            print(f"[ERROR] Directory not found: {burst_list_path}")
            raise e

        if split is None and seq_ids is not None:
            burst_list = [b for b in burst_list if b[:4] in seq_ids]
            print(f"[DEBUG] Filtered burst list by seq_ids: {burst_list}")
        elif split is not None and split != 'val_mine':
            lispr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            txt_path = '{}/data_specs/burstsr_{}.txt'.format(lispr_path, split)
            print(f"[DEBUG] Loading sequence IDs from: {txt_path}")
            try:
                seq_ids = load_txt(txt_path)
                burst_list = [b for b in burst_list if b[:4] in seq_ids]
                print(f"[DEBUG] Filtered burst list by text file sequence IDs: {burst_list}")
            except FileNotFoundError as e:
                print(f"[ERROR] Sequence ID file not found: {txt_path}")
                raise e

        return burst_list

    def get_burst_info(self, burst_id):
        burst_info = {'burst_size': 14, 'burst_name': self.burst_list[burst_id]}
        return burst_info

    def _get_raw_image(self, burst_id, im_id):
        """
        Здесь заменяем SamsungRAWImage на QoocamPNGImage.
        """
        # Исправляем путь, чтобы не было дублирования 'im_raw.png'
        image_path = '{}/{}/{}/im_raw.png'.format(self.root, self.split, self.burst_list[burst_id], im_id)
        raw_image = QoocamPNGImage.load(image_path)  # Загружаем изображение с помощью нового класса
        return raw_image

    def _get_gt_image(self, burst_id):
        """
        CanonImage оставим без изменений, поскольку это целевой ground truth.
        """
        canon_im = CanonImage.load(os.path.join(self.root, self.split, self.burst_list[burst_id], 'canon'))
        return canon_im

    def get_burst(self, burst_id, im_ids, info=None):
        frames = [self._get_raw_image(burst_id, i) for i in im_ids]

        gt = self._get_gt_image(burst_id)
        if info is None:
            info = self.get_burst_info(burst_id)

        return frames, gt, info


def get_burstsr_val_set(limit=None, split='val'):
    """ Get the BurstSR validation dataset """
    burstsr_dataset = BurstSRDataset(split=split, initialize=True)
    processing_fn = processing.BurstSRProcessing(transform=None, random_flip=False,
                                                 substract_black_level=True,
                                                 crop_sz=80)
    # Train sampler and loader
    dataset = sampler.IndexedBurst(burstsr_dataset, burst_size=14, processing=processing_fn)

    if limit is not None:
        dataset = dataset[:limit]

    return dataset
