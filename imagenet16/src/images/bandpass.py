import torch

from .lowpass import GaussianBlurAll

def apply_bandpass(images: torch.Tensor, s1: float, s2: float):
    """Apply band-pass filter to input images
    Args:
        images (torch.Tensor): (N, C, H, W)
        s1 (float): sigma1
        s2 (float): sigma2

    Returns: band-passed images
    """
    low1 = GaussianBlurAll(images, sigma=s1)
    if s2 == None:
        return low1
    else:
        low2 = GaussianBlurAll(images, sigma=s2)
        return low1 - low2