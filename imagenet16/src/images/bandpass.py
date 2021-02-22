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
