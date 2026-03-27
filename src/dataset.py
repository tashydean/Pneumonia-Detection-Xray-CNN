"""
dataset.py
ChestXRayDataset + DataLoader 팩토리.

데이터 구조 (Kaggle chest-xray-pneumonia):
  chest_xray/
    train/  NORMAL/  PNEUMONIA/
    val/    NORMAL/  PNEUMONIA/
    test/   NORMAL/  PNEUMONIA/
"""

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ChestXRayDataset(Dataset):
    """
    NORMAL=0, PNEUMONIA=1 이진 레이블 Dataset.
    이미지를 RGB로 읽어 pre_conv의 3채널 입력과 일치시킴.
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                # .jpg / .jpeg / .png 모두 포함
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')  # 그레이스케일 아닌 RGB
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    train / val / test DataLoader 반환.
    """
    train_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), get_train_transforms())
    val_dataset   = ChestXRayDataset(os.path.join(data_dir, 'val'),   get_eval_transforms())
    test_dataset  = ChestXRayDataset(os.path.join(data_dir, 'test'),  get_eval_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[Dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader
