import os
import pathlib
import sys
from typing import Dict, Optional, List

import torch

# add a path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.dataset.imagenet16 import load_data, num_classes
from src.images.lowpass import GaussianBlurAll


def make_bandpass_images(
    target_id: int = 1,
    num_filters: int = 6,
    num_images: int = 10,
) -> torch.Tensor:
    """Return test images consisting of (num_images) images for each class.
    Arguments:
        target_id (int): label id of the target category. Default: 1
        num_images (int): number of images for each class. Default: 10

    Returns: images (torch.Tensor)
        shape: (num_classes*num_images, 3, 244, 244)
            where:
                num_classes = 16
    """
    _, test_loader = load_data(batch_size=32)

    counts = torch.zeros(num_classes)
    test_images = torch.zeros([num_classes, num_images, 3, 224, 224])
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            label_id = label.item()
            if counts[label_id] < 10:
                test_images[label_id][int(counts[label_id])] = image
                counts[label_id] += 1

    new_test_images = torch.zeros([num_filters + 1, num_images, 3, 224, 224])

    new_test_images[0] = test_images[target_id]  # add raw images

    # bandpass images
    filters = make_bandpass_filters(num_filters=num_filters)
    for i, (s1, s2) in enumerate(filters.values(), 1):
        low1 = GaussianBlurAll(test_images[target_id], sigma=s1)
        if s2 == None:
            new_test_images[i] = low1
        else:
            low2 = GaussianBlurAll(test_images[target_id], sigma=s2)
            new_test_images[i] = low1 - low2

    # reshape to (N, C, H, W)
    # new_test_images = new_test_images.view(-1, *test_images.shape[2:])
    return new_test_images


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
