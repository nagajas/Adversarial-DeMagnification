import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np

import uuid
from albumentations.augmentations.crops.transforms import Crop
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate
from skimage.metrics import structural_similarity
import albumentations as A
from transforms.albu import IsotropicResize

class DeepFakesDataset(Dataset):
    def __init__(self, images_paths, labels, image_size = 224, mode='train', additional_path = []):
        self.x = images_paths
        self.y = labels
        self.n_samples = len(images_paths)
        self.mode = mode
        self.image_size = image_size
        self.additional_path = additional_path

    # def create_train_transforms(self, size):
    #     if size <=0 :
    #         print("Size error!!!")
    #     return Compose([
    #         # OneOf([
    #         #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
    #         #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
    #         #     IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
    #         # ], p=0.99),
    #         IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),  # Single Resize transform
    #         PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    #         ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
    #         GaussNoise(p=0.3),
    #         #GaussianBlur(blur_limit=3, p=0.05),
    #         HorizontalFlip(),
    #         ToGray(p=0.2),
    #         ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    #     ]
    #     )
    def create_train_transforms(self, size):
        if size <= 0:
            raise ValueError("Image size must be greater than 0.")

        return A.Compose([
            A.Resize(size, size),  # Ensures all images have the same shape
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            A.ImageCompression(quality_range=(80,100), p=0.2),
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(),
            A.ToGray(p=0.2),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-5, 5), p=0.5),  # Replaces ShiftScaleRotate
        ])
    # def create_val_transforms(self, size):
    #     return Compose([
    #         IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
    #         PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    #     ])
    def create_val_transforms(self, size):
        return A.Compose([
            A.Resize(size, size),  # Ensures all validation images are the same size
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])
    
    def __getitem__(self, index):
        image_path = self.x[index]
        #print(f"Image size: {self.image_size}") 
        image = cv2.imread(image_path)


        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transforms(self.image_size)

        image = transform(image=image)["image"]
        label = self.y[index]
        if len(self.additional_path) > 0 and self.mode != 'train':
            additional_image_path = image_path.replace(self.additional_path[0], self.additional_path[1])
            additional_image = cv2.imread(additional_image_path, 1)
            additional_image = transform(image=additional_image)["image"]
            
            (score, diff) = structural_similarity(image, additional_image, full=True, multichannel=True)

            return torch.tensor(image, dtype=torch.float), torch.tensor(label), image_path, torch.tensor(additional_image, dtype=torch.float), additional_image_path, score
        else:
            return torch.tensor(image, dtype=torch.float), torch.tensor(label), image_path

    
    def __len__(self):
        return self.n_samples
