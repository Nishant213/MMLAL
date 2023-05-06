import os
import os.path as osp

import random
import numpy as np
from PIL import Image, ImageFilter

import torch
from torchvision import transforms
import torchvision.transforms.functional as transforms_f

import cv2
import matplotlib.pyplot as plt

#Ref: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/transforms.py
try:
    from torchvision.transforms.functional import InterpolationMode
    map_nearest = InterpolationMode.NEAREST
    map_bilinear = InterpolationMode.BILINEAR
    map_bicubic = InterpolationMode.BICUBIC
except ImportError:
    map_nearest = Image.NEAREST
    map_bilinear = Image.BILINEAR
    map_bicubic = Image.BICUBIC

# #Ref: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
def CutOut(image, label, pseudo_label = None, n_holes = 1, sz_holes = 50, img_pad = 0, label_pad = 255):
        image = np.asarray(image)
        label = np.asarray(label)
        
        if pseudo_label is not None:
            pseudo_label = np.asarray(pseudo_label)
    
        w = image.shape[0]
        h = image.shape[1]

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            x1 = np.clip(y - sz_holes // 2, 0, h)
            x2 = np.clip(y + sz_holes // 2, 0, h)
            y1 = np.clip(x - sz_holes // 2, 0, w)
            y2 = np.clip(x + sz_holes // 2, 0, w)

            image[y1: y2, x1: x2] = img_pad
            label[y1: y2, x1: x2] = label_pad
            if pseudo_label is not None:
                pseudo_label[y1: y2, x1: x2] = img_pad
            
        image = Image.fromarray(image.astype(np.uint8))
        label = Image.fromarray(label.astype(np.uint8))
        if pseudo_label is not None:
            pseudo_label = Image.fromarray(pseudo_label.astype(np.uint8))
        
        if pseudo_label is not None:
            return image, label, pseudo_label
        else:
            return image, label, None

def do_transforms(image,
                  label,
                  pseudo_label = None,
                  ratio_range = (1.0, 1.0),
                  crop_size = (224, 224),
                  use_augmentation_weak=True,
                  use_augmentation_strong=True,
                  use_augmentation_cutout=True):
    
    og_w, og_h = image.size
    
    scale_ratio = random.uniform(ratio_range[0], ratio_range[1])
    
    resized_image_size = (int(og_h * scale_ratio), int(og_w * scale_ratio))
    image = transforms_f.resize(image, resized_image_size, map_bilinear)
    label = transforms_f.resize(label, resized_image_size, map_nearest)
    
    if pseudo_label is not None:
        pseudo_label  = transforms_f.resize(pseudo_label, resized_image_size, map_nearest)
    
    if crop_size[0] > resized_image_size[0] or crop_size[1] > resized_image_size[1]:
        
        pad_lr = max(crop_size[1] - resized_image_size[1], 0)
        pad_hw = max(crop_size[0] - resized_image_size[0], 0)
        right_pad, bottom_pad = int(pad_lr * 0.5), int(pad_hw * 0.5)
            
        image = transforms_f.pad(image, padding=(pad_lr - right_pad, pad_hw - bottom_pad, right_pad, bottom_pad), fill=0)
        label = transforms_f.pad(label, padding=(pad_lr - right_pad, pad_hw - bottom_pad, right_pad, bottom_pad), fill=255)
        
        if pseudo_label is not None:
            pseudo_label = transforms_f.pad(pseudo_label, padding=(pad_lr - right_pad, pad_hw - bottom_pad, right_pad, bottom_pad), fill=255)
    
    if ratio_range[0] != 1.0:
        # Random Cropping
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
        image = transforms_f.crop(image, i, j, h, w)
        label = transforms_f.crop(label, i, j, h, w)
        if pseudo_label is not None:
            pseudo_label = transforms_f.crop(pseudo_label, i, j, h, w)
    else:
        image = transforms_f.center_crop(image, crop_size)
        label = transforms_f.center_crop(label, crop_size)
        if pseudo_label is not None:
            pseudo_label = transforms_f.center_crop(pseudo_label, crop_size)
    
    if use_augmentation_weak:
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if pseudo_label is not None:
                pseudo_label = transforms_f.hflip(pseudo_label)
            
    if use_augmentation_strong:
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.25, 1.25)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
            
        elif torch.rand(1) > 0.5:
            color_transforms_params = transforms.ColorJitter(brightness=0.8,
                                                             contrast=0.8,
                                                             saturation=0.8,
                                                             hue=0.2
                                                             )
            image = color_transforms_params(image)
            
            if torch.rand(1) > 0.25:
                image = transforms.Grayscale(num_output_channels=3)(image)
            
    if use_augmentation_cutout:
        if torch.rand(1) > 0.5:
            image, label, pseudo_label = CutOut(image, label, pseudo_label)
        
    return image, label, pseudo_label
