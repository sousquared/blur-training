import itertools
from typing import Dict, Optional, List

import torch

from .lowpass import GaussianBlurAll


def apply_bandpass_filter(images: torch.Tensor, s1: float, s2: float) -> torch.Tensor:
    """Apply band-pass filter to input images
    Args:
        images (torch.Tensor): (N, C, H, W)
        s1 (float): sigma1
        s2 (float): sigma2

    Returns (torch.Tensor): band-passed images (N, C, H, W)
    """
    low1 = GaussianBlurAll(images, sigma=s1)
    if s2 == None:
        return low1
    else:
        low2 = GaussianBlurAll(images, sigma=s2)
        return low1 - low2


def make_bandpass_filters(
    num_filters: int = 6,
) -> Dict[int, Optional[List[int]]]:
    filters = {}
    filters[0] = [0, 1]
    for i in range(1, num_filters):
        if i == (num_filters - 1):
            filters[i] = [2 ** (i - 1), None]  # last band-pass is low-pass filter
        else:
            filters[i] = [2 ** (i - 1), 2 ** i]

    return filters


def make_filter_combinations(filters: dict) -> list:
    """Makes all combinations from filters
    Args:
        filters (dict): Dict[int, List[int, int]]

    Returns (list): all combinations of filter id
    """
    filter_comb = []
    for i in range(1, len(filters)):
        filter_comb += list(itertools.combinations(filters.keys(), i))

    return filter_comb
