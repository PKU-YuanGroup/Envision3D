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


def add_margin(pil_img, color=0, size=256):
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


class EightAnchorImageDataset(Dataset):
    def __init__(self,
                 image_paths,
                 scene,
                 img_wh: Tuple[int, int],
                 bg_color: str,
                 crop_size: int = 224) -> None:
        """
        Create a dataset from three single images.
        """
        self.image_paths = image_paths
        self.img_wh = img_wh
        self.crop_size = -1
        self.bg_color = bg_color
        # self.sequences = [[0, 1, 2]]
        self.sequences = [[7, 0, 1], [1, 2, 3], [3, 4, 5], [5, 6, 7]]
        # Load and process the three images
        bg_color = self.get_bg_color()
        self.all_images = []

        for i in range(8):
            path = os.path.join(self.image_paths, f"{scene}", f"rgb{i*4}.png")
            img = self.load_image(path, bg_color)
            self.all_images.append(img.permute(2, 0, 1))

    # def __len__(self):
    #     return len(self.all_images)

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

    def load_image(self, img_path, bg_color, return_type='pt', Imagefile=None):
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
            pass
            image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
            image_input = image_input.resize((image_size, image_size))

        # img = scale_and_place_object(img, self.scale_ratio)
        img = np.array(image_input)
        img = img.astype(np.float32) / 255.  # [0, 1]

        if img.shape[-1] == 4:
            alpha = img[..., 3:4]
            img = img[..., :3] * alpha + bg_color * (1 - alpha)
        elif img.shape[-1] == 3:
            img = img

        if return_type == "np":
            pass
        elif return_type == "pt":
            img = torch.from_numpy(img)
            # alpha = torch.from_numpy(alpha)
        else:
            raise NotImplementedError

        return img

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_indices = self.sequences[index % len(self.sequences)]
        imgs = [self.all_images[i] for i in sequence_indices]

        out = {
            'imgs_in': imgs,
            "seq": sequence_indices,
        }

        return out

