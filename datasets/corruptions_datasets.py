
import os
import json
import torch
import logging
from glob import glob
from typing import Optional, Sequence

from robustbench.data import CORRUPTIONS, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder

logger = logging.getLogger(__name__)

def create_imagenetc_dataset(
    n_examples: Optional[int] = -1,
    severity: int = 5,
    data_dir: str = './data',
    corruption: str = "gaussian_noise",
    corruptions_seq: Sequence[str] = CORRUPTIONS,
    transform=None,
    setting: str = 'continual'):

    # create the dataset which loads the default test list from robust bench containing 5000 test samples
    corruptions_seq = corruptions_seq if "mixed_domains" in setting else [corruption]
    corruption_dir_path = os.path.join(data_dir, corruptions_seq[0], str(severity))
    dataset_test = CustomImageFolder(corruption_dir_path, transform)

    if "mixed_domains" in setting or "correlated" in setting or n_examples != -1:
        # load imagenet class to id mapping from robustbench
        with open(os.path.join("robustbench", "data", "imagenet_class_to_id_map.json"), 'r') as f:
            class_to_idx = json.load(f)

        if n_examples != -1 or "correlated" in setting:
            # create file path of file containing all 50k image ids
            file_path = os.path.join("datasets", "imagenet_list", "imagenet_val_ids_50k.txt")
        else:
            # create file path of default test list from robustbench
            file_path = os.path.join("robustbench", "data", "imagenet_test_image_ids.txt")

        # load file containing file ids
        with open(file_path, 'r') as f:
            fnames = f.readlines()

        item_list = []
        for cor in corruptions_seq:
            corruption_dir_path = os.path.join(data_dir, cor, str(severity))
            item_list += [(os.path.join(corruption_dir_path, fn.split('\n')[0]), class_to_idx[fn.split(os.sep)[0]]) for fn in fnames]
        dataset_test.samples = item_list

    return dataset_test
