import copy
import inspect
import random
import sys

import torch
from torch.utils.data.dataset import random_split

from src.datasets.cars import Cars
from src.datasets.dtd import DTD
from src.datasets.eurosat import EuroSAT, EuroSATVal
from src.datasets.gtsrb import GTSRB
from src.datasets.mnist import MNIST
from src.datasets.resisc45 import RESISC45
from src.datasets.sun397 import SUN397
from src.datasets.svhn import SVHN

registry = {name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset,
    new_dataset_class_name: str,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    max_val_samples=None,
    seed: int = 0,
):
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed))
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers)

    new_dataset.test_loader_shuffle = torch.utils.data.DataLoader(
        new_dataset.test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name: str,
    preprocess,
    location: str,
    batch_size: int = 128,
    num_workers: int = 0,
    val_fraction: float = 0.1,
    max_val_samples: int = 5000,
):
    """
    Returns a dataset object for the given dataset name.

    Args:
        dataset_name (str): Name of the dataset.
        preprocess: Preprocessing function to apply to the dataset.
        location (str): Location of the dataset.
        batch_size (int, optional): Batch size for the dataset. Defaults to 128.
        num_workers (int, optional): Number of workers for the dataset. Defaults to 0.
        val_fraction (float, optional): Fraction of the dataset to use for validation. Defaults to 0.1.
        max_val_samples (int, optional): Maximum number of validation samples to use. Defaults to 5000.

    Returns:
        dataset: A dataset object for the given dataset name.
    """
    if dataset_name.endswith("Val"):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(base_dataset_name, preprocess, location, batch_size, num_workers)
            dataset = split_train_into_train_val(base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples)
            return dataset
    else:
        assert dataset_name in registry, f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
        dataset_class = registry[dataset_name]
    dataset = dataset_class(preprocess, location=location, batch_size=batch_size, num_workers=num_workers)
    return dataset
