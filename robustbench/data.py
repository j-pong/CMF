import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from robustbench.model_zoo.enums import BenchmarkDataset
from robustbench.loaders import CustomImageFolder


PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}


def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor


def load_imagenet(
        n_examples: Optional[int] = 5000,
        data_dir: str = './data',
        prepr: str = 'Res256Crop224') -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]
    imagenet = CustomImageFolder(data_dir + '/val', transforms_test)
    
    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=False, num_workers=4)

    x_test, y_test, paths = next(iter(test_loader))
    
    return x_test, y_test


CleanDatasetLoader = Callable[[Optional[int], str], Tuple[torch.Tensor,
                                                          torch.Tensor]]
_clean_dataset_loaders: Dict[BenchmarkDataset, CleanDatasetLoader] = {
    BenchmarkDataset.imagenet: load_imagenet,
}


def load_clean_dataset(dataset: BenchmarkDataset, n_examples: Optional[int],
                       data_dir: str, prepr: Optional[str] = 'none') -> Tuple[torch.Tensor, torch.Tensor]:
    return _clean_dataset_loaders[dataset](n_examples, data_dir, prepr)


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.imagenet: "ImageNet-C"
}

def load_imagenetc(
        n_examples: Optional[int] = 5000,
        severity: int = 5,
        data_dir: str = './data',
        shuffle: bool = False,
        corruptions: Sequence[str] = CORRUPTIONS,
        prepr: str = 'Res256Crop224'
) -> Tuple[torch.Tensor, torch.Tensor]:
    transforms_test = PREPROCESSINGS[prepr]

    assert len(corruptions) == 1, "so far only one corruption is supported (that's how this function is called in eval.py"

    data_folder_path = Path(data_dir) / CORRUPTIONS_DIR_NAMES[BenchmarkDataset.imagenet] / corruptions[0] / str(severity)
    imagenet = CustomImageFolder(data_folder_path, transforms_test)

    test_loader = data.DataLoader(imagenet, batch_size=n_examples,
                                  shuffle=shuffle, num_workers=2)

    x_test, y_test, paths = next(iter(test_loader))

    return x_test, y_test


CorruptDatasetLoader = Callable[[int, int, str, bool, Sequence[str]],
                                Tuple[torch.Tensor, torch.Tensor]]
CORRUPTION_DATASET_LOADERS: Dict[BenchmarkDataset, CorruptDatasetLoader] = {
    BenchmarkDataset.imagenet: load_imagenetc,
}