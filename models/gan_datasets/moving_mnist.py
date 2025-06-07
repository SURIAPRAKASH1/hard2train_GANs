import torch
from torch.utils.data import Dataset
import torch.nn as nn 
import torchvision
from torchvision.transforms import v2

from typing import Type


class CustomNormalizer(nn.Module):
    """
    CustomNormalizer to normalize image pixels to within range of (-1, 1) according to paper.
    cause we are using tanh in Generator Model at last layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, C, H, W = x.shape       # (channels, height, width)

        # if grey scale image then we have to make it have 3 pixels
        if C < 3:
            x = torch.repeat_interleave(x, dim = 1, repeats = 3)

        x = (x / 255) * 2 - 1      # normalizer to (-1, 1) range
        return x.transpose(0, 1)   # (C, T, H, W)


# pipeline
transform = v2.Compose([
    v2.ToDtype(dtype= torch.float32),
    CustomNormalizer(),
])

# Dataset
def get_mmnist_dataset(root: str) -> Dataset:
    return torchvision.datasets.MovingMNIST(root, download= True, transform= transform)