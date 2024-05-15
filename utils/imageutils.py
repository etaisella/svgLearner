from math import ceil
import torch
from torch.nn.functional import conv2d
from torch.distributions import Normal

def boxBlurfilter(in_channels: int=1, out_channels: int=1, kernel_size: int=7):
    box_filter = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
    box_filter.weight.data = torch.ones(in_channels, out_channels, kernel_size, kernel_size).float() / (kernel_size ** 2)
    box_filter.weight.requires_grad = False
    return box_filter
    