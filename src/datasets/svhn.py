import os

import numpy as np
import torch
from torchvision.datasets import SVHN as PyTorchSVHN


class SVHN:
    """
    A class representing the Street View House Numbers (SVHN) dataset.

    Args:
        preprocess (callable): a function/transform that takes in an PIL image and returns a transformed version
        location (str, optional): the directory to store the dataset (default: "~/data")
        batch_size (int, optional): how many samples per batch to load (default: 128)
        num_workers (int, optional): how many subprocesses to use for data loading (default: 0)

    Attributes:
        train_dataset (torch.utils.data.Dataset): the training dataset
        train_loader (torch.utils.data.DataLoader): a data loader for the training dataset
        test_dataset (torch.utils.data.Dataset): the test dataset
        test_loader (torch.utils.data.DataLoader): a data loader for the test dataset
        test_loader_shuffle (torch.utils.data.DataLoader): a shuffled data loader for the test dataset
        classnames (list): a list of class names

    Example:
        >>> preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        >>> svhn = SVHN(preprocess)
    """

    def __init__(
        self,
        preprocess,
        location: str = os.path.expanduser("~/data"),
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        # to fit with repo conventions for location
        modified_location = os.path.join(location, "svhn")

        self.train_dataset = PyTorchSVHN(root=modified_location, download=True, split="train", transform=preprocess)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = PyTorchSVHN(root=modified_location, download=True, split="test", transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_shuffle = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        self.classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
