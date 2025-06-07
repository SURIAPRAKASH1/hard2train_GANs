import torch
from torch.utils.data import Dataset
import torch.nn as nn 
import torchvision
from torchvision.transforms import v2

from typing import Type


###### celebA dataset preprocessing #######

class CustomNormalizer(nn.Module):
    """
    CustomNormalizer to normalize image pixels to within range of (-1, 1) according to paper. 
    cause we are using tanh in Generator Model at last layer
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C, H, W = x.shape       # (channels, height, width)

        # if grey scale image then we have to make it have 3 pixels
        if C < 3:
            x = torch.repeat_interleave(x, dim = 0, repeats = 3)

        x = (x / 255) * 2 - 1      # normalizer to (-1, 1) range
        return x


transform = v2.Compose([
    v2.ToImage(),                            # converts Image to tensor
    v2.Resize((64, 64), antialias= True),    # Resizeing . (v2.Resize() only work with tensor correctly)
    v2.ToDtype(dtype= torch.float32),        # sanity check
    CustomNormalizer(),                      # custom normalizer do's some transformation   
])


# Dataset
celebA_root: str = "https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing"
celebA_dataset: Type[Dataset] = torchvision.datasets.CelebA(celebA_root, download= True, transform= transform)