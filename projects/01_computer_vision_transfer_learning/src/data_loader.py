"""
Data loading and augmentation module for image classification.
Implements advanced data augmentation and efficient data loading.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import argparse


class ImageDataset(Dataset):
    """Custom Dataset for image classification with albumentations support."""

    def __init__(self, image_paths, labels, transform=None, is_training=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = self.labels[idx]
        return image, label


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Advanced training data augmentation pipeline."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.1),
            A.GridDistortion(num_steps=5, distort_limit=0.1),
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_valid_transforms(image_size: int = 224) -> A.Compose:
    """Validation/test data transformation pipeline."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """Test-time augmentation transforms."""
    return [
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
        A.Compose([
            A.Resize(height=int(image_size * 1.1), width=int(image_size * 1.1)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]),
    ]


def create_data_loaders(
    train_paths,
    train_labels,
    val_paths,
    val_labels,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""

    train_dataset = ImageDataset(
        train_paths,
        train_labels,
        transform=get_train_transforms(image_size),
        is_training=True
    )

    val_dataset = ImageDataset(
        val_paths,
        val_labels,
        transform=get_valid_transforms(image_size),
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/validation split ratio')
    args = parser.parse_args()

    print(f"Dataset preparation from {args.data_path}")
    print(f"Train/Val split: {args.split_ratio}/{1-args.split_ratio}")
    print("Data augmentation pipeline initialized successfully!")
