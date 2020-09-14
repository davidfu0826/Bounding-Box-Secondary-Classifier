import os
import warnings
from typing import List
warnings.simplefilter("ignore", FutureWarning)

import PIL.Image as Image
from PIL.Image import BICUBIC
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, Lambda
from torchvision.transforms import ColorJitter, RandomAffine, RandomPerspective, RandomRotation, RandomErasing, RandomCrop, Grayscale
from torchvision.transforms import RandomChoice, RandomApply
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


class CustomImageDataset(Dataset):
    """Custom Dataset for loading images from paths"""

    def __init__(self, img_paths, transform=None, label_names=None):
    
        self.img_paths = img_paths
        self.y = [os.path.basename(os.path.dirname(img_path)) for img_path in self.img_paths]
        self.class_to_idx = {label:idx for idx, label in enumerate(label_names)}
        self.y = [self.class_to_idx[label] for label in self.y]
        
        self.transform = transform

    def __getitem__(self, index):
        #if torch.is_tensor(index):
        #    idx = idx.tolist()
            
        img = Image.open(self.img_paths[index])
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.y)
    
def get_test_transforms(img_size: int) -> Compose:
    """Returns data transformations for test dataset.
    
    Args:
        img_size: The resolution of the input image (img_size x img_size)
    """
    return Compose([
        Resize([img_size, img_size]),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
def get_train_transforms(img_size: int) -> Compose:
    """Returns data transformations/augmentations for train dataset.
    
    Args:
        img_size: The resolution of the input image (img_size x img_size)
    """
    return Compose([
        RandomApply([
            ColorJitter(brightness=0.3, contrast=0.01, saturation=0.01, hue=0),
            RandomAffine(0.1, translate=(0.04,0.04), scale=(0.04,0.04), shear=0.01, resample=2),
            #Grayscale(num_output_channels=3),
            #RandomCrop(30),
            RandomPerspective(0.1)
        ]),
        Resize([img_size, img_size], interpolation=3),
        ToTensor(),
        #RandomApply([    
            #RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        #]),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
def undersample(img_paths: List[str]) -> List[str]:
    y = list()
    for img_path in img_paths:
        dir_path = os.path.dirname(img_path)
        label = os.path.basename(dir_path)
        y.append(label)
        
    img_paths = [[a] for a in img_paths] # Correct format
    sampler = RandomUnderSampler()
    new_img_paths, _ = sampler.fit_resample(img_paths, y)
    new_img_paths = [a[0] for a in new_img_paths]

    return new_img_paths

def oversample(img_paths: List[str]) -> List[str]:
    y = list()
    for img_path in img_paths:
        dir_path = os.path.dirname(img_path)
        label = os.path.basename(dir_path)
        y.append(label)
        
    img_paths = [[a] for a in img_paths] # Correct format
    sampler = RandomOverSampler()
    new_img_paths, _ = sampler.fit_resample(img_paths, y)
    new_img_paths = [a[0] for a in new_img_paths]

    return new_img_paths
