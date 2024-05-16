from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

from glob import glob

import PIL.Image

import pdb

import cv2
import numpy as np


def add_margin(pil_img, color=(255,255,255), size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def scale_and_place_object(image, scale_factor):
    assert np.shape(image)[-1] == 4  # RGBA

    # Extract the alpha channel (transparency) and the object (RGB channels)
    alpha_channel = image[:, :, 3]

    # Find the bounding box coordinates of the object
    coords = cv2.findNonZero(alpha_channel)
    x, y, width, height = cv2.boundingRect(coords)

    # Calculate the scale factor for resizing
    original_height, original_width = image.shape[:2]

    if width > height:
        size = width
        original_size = original_width
    else:
        size = height
        original_size = original_height

    scale_factor = min(scale_factor, size / (original_size + 0.0))

    new_size = scale_factor * original_size
    scale_factor = new_size / size

    # Calculate the new size based on the scale factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    center_x = original_width // 2
    center_y = original_height // 2

    paste_x = center_x - (new_width // 2)
    paste_y = center_y - (new_height // 2)

    # Resize the object (RGB channels) to the new size
    rescaled_object = cv2.resize(image[y:y + height, x:x + width], (new_width, new_height))

    # Create a new RGBA image with the resized image
    new_image = np.zeros((original_height, original_width, 4), dtype=np.uint8)

    new_image[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = rescaled_object

    return new_image


class SingleImageDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 num_views: int,
                 img_wh: Tuple[int, int],
                 bg_color: str,
                 crop_size: int = 224,
                 single_image: Optional[PIL.Image.Image] = None,
                 num_validation_samples: Optional[int] = None,
                 filepaths: Optional[list] = None,
                 cond_type: Optional[str] = None
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type

        if single_image is None:
            if filepaths is None:
                # Get a list of all files in the directory
                file_list = os.listdir(self.root_dir)
            else:
                file_list = filepaths

            # Filter the files that end with .png or .jpg
            self.file_list = [file for file in file_list if file.endswith(('.png', '.jpg'))]
        else:
            self.file_list = None

        # load all images
        self.all_images = []
        self.all_alphas = []
        bg_color = self.get_bg_color()

        if single_image is not None:
            image, alpha = self.load_image(None, bg_color, return_type='pt', Imagefile=single_image)
            self.all_images.append(image)
            self.all_alphas.append(alpha)
        else:
            for file in self.file_list:
                print(os.path.join(self.root_dir, file))
                image, alpha = self.load_image(os.path.join(self.root_dir, file), bg_color, return_type='pt')
                self.all_images.append(image)
                self.all_alphas.append(alpha)

        self.all_images = self.all_images[:num_validation_samples]
        self.all_alphas = self.all_alphas[:num_validation_samples]

    def __len__(self):
        return len(self.all_images)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None):
        # pil always returns uint8
        if Imagefile is None:
            image_input = Image.open(img_path)
        else:
            image_input = Imagefile
        image_size = self.img_wh[0]

        if self.crop_size != -1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
        else:
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError

        return img, alpha

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image = self.all_images[index % len(self.all_images)]
        alpha = self.all_alphas[index % len(self.all_images)]
        if self.file_list is not None:
            filename = self.file_list[index % len(self.all_images)].replace(".png", "")
        else:
            filename = 'null'

        img_tensors_in = [
                             image.permute(2, 0, 1)
                         ] * self.num_views

        alpha_tensors_in = [
                               alpha.permute(2, 0, 1)
                           ] * self.num_views


        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()  # (Nv, 3, H, W)
        alpha_tensors_in = torch.stack(alpha_tensors_in, dim=0).float()  # (Nv, 3, H, W)


        out = {
            'imgs_in': img_tensors_in,
            'alphas': alpha_tensors_in,
            'filename': filename,
        }

        return out

class SingleImageNormalDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 img_wh: Tuple[int, int],
                 bg_color: str,
                 num_views: int = 8,
                 crop_size: int = 224,
                 single_image: Optional[PIL.Image.Image] = None,
                 single_normal: Optional[PIL.Image.Image] = None,
                 num_validation_samples: Optional[int] = None,
                 filepaths: Optional[list] = None,
                 cond_type: Optional[str] = None
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.num_views = num_views
        self.img_wh = img_wh
        self.crop_size = crop_size
        self.bg_color = bg_color
        self.cond_type = cond_type

        if single_image is None:
            if filepaths is None:
                # Get a list of all files in the directory
                file_list = os.listdir(self.root_dir)
            else:
                file_list = filepaths

            # Filter the files that end with .png or .jpg
            self.file_list = [file for file in file_list if file.endswith(('.png', '.jpg'))]
        else:
            self.file_list = None

        # load all images
        self.all_images = []
        self.all_alphas = []
        self.all_normals = []
        bg_color = self.get_bg_color()

        if single_image is not None:
            image, alpha, normal = self.load_image(single_image, bg_color, return_type='pt')
            self.all_images.append(image)
            self.all_alphas.append(alpha)
            self.all_normals.append(normal)
        else:
            for file in self.file_list:
                print(os.path.join(self.root_dir, 'rgb', file))
                image, alpha, normal = self.load_image(os.path.join(self.root_dir, file), bg_color, return_type='pt')
                self.all_images.append(image)
                self.all_alphas.append(alpha)
                self.all_normals.append(normal)

        self.all_images = self.all_images[:num_validation_samples]
        self.all_alphas = self.all_alphas[:num_validation_samples]
        self.all_normals = self.all_normals[:num_validation_samples]

    def __len__(self):
        return len(self.all_images)

    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(self, img_path, bg_color, return_type='np', Imagefile=None, Normalfile=None):
        # pil always returns uint8
        if Imagefile is None:

            image_input = Image.open(img_path)
            _, file_name = os.path.split(img_path)
            name, _ = os.path.splitext(file_name)
            normal_input = Image.open(os.path.join(self.root_dir,'normal', f"{name}_normal.png"))
        else:
            image_input = Imagefile
            normal_input = Normalfile
        print(image_input.height, normal_input.height)
        assert image_input.height == normal_input.height
        assert image_input.width == normal_input.width
        image_size = self.img_wh[0]

        if self.crop_size != -1:
            alpha_np = np.asarray(image_input)[:, :, 3]
            coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
            min_x, min_y = np.min(coords, 0)
            max_x, max_y = np.max(coords, 0)
            ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
            ref_nor_ = normal_input.crop((min_x, min_y, max_x, max_y))
            h, w = ref_img_.height, ref_img_.width
            scale = self.crop_size / max(h, w)
            h_, w_ = int(scale * h), int(scale * w)
            ref_img_ = ref_img_.resize((w_, h_))
            ref_nor_ = ref_nor_.resize((w_, h_))
            image_input = add_margin(ref_img_, size=image_size)
            normal_input = add_margin(ref_nor_, size=image_size)
        else:
            pass
            # image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            # image_input = image_input.resize((image_size, image_size))
            # normal_input = add_margin(normal_input, size=max(image_input.height, image_input.width))
            # normal_input = normal_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255.  # [0, 1]
        normal = np.array(normal_input)
        normal = normal.astype(np.float32) / 255.  # [0, 1]
        assert img.shape[-1] == 4  # RGBA

        alpha = img[..., 3:4]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            normal = torch.from_numpy(normal)
            alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError

        return img, alpha, normal

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        image = self.all_images[index % len(self.all_images)]
        alpha = self.all_alphas[index % len(self.all_images)]
        normal = self.all_normals[index % len(self.all_images)]

        if self.file_list is not None:
            filename = self.file_list[index % len(self.all_images)].replace(".png", "")
        else:
            filename = 'null'

        img_tensors_in = image.permute(2, 0, 1).float()

        alpha_tensors_in = alpha.permute(2, 0, 1).float()

        normal_tensors_in = normal.permute(2, 0, 1).float()



        out = {
            'imgs_in': img_tensors_in,
            'normals_in': normal_tensors_in,
            'alphas': alpha_tensors_in,
            'filename': filename,
        }

        return out

# from tqdm import tqdm
# validation_dataset = SingleImageNormalDataset(root_dir='./validation_ref/', num_views=9, bg_color="white",
#                                         crop_size=224,num_validation_samples=1000, img_wh=[256, 256])
# validation_dataloader = torch.utils.data.DataLoader(
#     validation_dataset, batch_size=1, shuffle=False, num_workers=8
# )
#
# for i, batch in tqdm(enumerate(validation_dataloader)):
#     # (1, v, 3, H, W)
#     bsz = batch["imgs_in"].shape[0]