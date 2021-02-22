import os
import pathlib
import sys
from typing import Dict, Optional, List
import itertools

import torch

# add a path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.dataset.imagenet16 import load_data, num_classes, num_channels, height, width
from src.images.bandpass import apply_bandpass_filter


def make_bandpass_images(
    target_id: int = 1,
    num_filters: int = 6,
    num_images: int = 10,
) -> torch.Tensor:
    """Makes band-passed test images (1 class).
    Arguments:
        target_id (int): label id of the target category. Default: 1
        num_filters (int): number of band-pass filters.
        num_images (int): number of images for each class. Default: 10

    Returns: images (torch.Tensor)
        shape: (num_images, num_filters + 1, C, H, W)
            where: num_classes = 16
    """
    # choose one class as test images
    raw_images = make_test_images_by_class(num_images=num_images)
    # choose one class
    raw_images = raw_images[target_id]  # (N, C, H, W)

    test_images = torch.zeros(
        [num_filters + 1, num_images, num_channels, height, width]
    )

    test_images[0] = raw_images  # add raw images

    # bandpass images
    filters = make_bandpass_filters(num_filters=num_filters)
    for i, (s1, s2) in enumerate(filters.values(), 1):
        test_images[i] = apply_bandpass_filter(images=raw_images, s1=s1, s2=s2)

    # reshape to (-1, C, H, W)
    # test_images = test_images.view(-1, *test_images.shape[2:])

    # change the order of num_images and num_filters(+1)
    return test_images.transpose(1, 0)


def make_bandpass_images_all_comb(
    target_id: int = 1,
    num_filters: int = 6,
    num_images: int = 10,
) -> torch.Tensor:
    """Makes band-passed test images (1 class) with all combinations of band-pass filters.
    Arguments:
        target_id (int): label id of the target category. Default: 1
        num_filters (int): number of band-pass filters.
        num_images (int): number of images for each class. Default: 10

    Returns: images (torch.Tensor)
        shape: (num_images, 2 ** num_filters - 1, C, H, W)
            where: num_classes = 16
    """
    # choose one class as test images
    raw_images = make_test_images_by_class(num_images=num_images)
    # choose one class
    raw_images = raw_images[target_id]  # (N, C, H, W)

    # make filter combinations
    filters = make_bandpass_filters(num_filters=num_filters)
    filter_comb = make_filter_combinations(filters=filters)

    test_images = torch.zeros(
        [len(filter_comb) + 1, num_images, num_channels, height, width]
    )

    test_images[0] = raw_images  # add raw images

    for i, filter_ids in enumerate(filter_comb, 1):
        img_list = []
        for filter_id in filter_ids:
            s1, s2 = filters[filter_id]
            img_list += [apply_bandpass_filter(images=raw_images, s1=s1, s2=s2)]

        if len(img_list) == 1:
            test_images[i] = img_list[0]
        else:
            # add all band-pass filters
            test_img = torch.zeros([num_images, num_channels, height, width])
            for img in img_list:
                test_img += img

            test_images[i] = test_img

    return test_images.transpose(1, 0)


def make_test_images_by_class(num_images: int = 10) -> torch.Tensor:
    """Makes test images along class labels.
    Args:
        num_images (int): number of images for each class. Default: 10

    Returns: test images (num_classes, N, C, H, W)
        where: num_classes = 16
    """
    _, test_loader = load_data(batch_size=32)

    counts = torch.zeros(num_classes)
    test_images = torch.zeros([num_classes, num_images, num_channels, height, width])
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            label_id = label.item()
            if counts[label_id] < num_images:
                test_images[label_id][int(counts[label_id])] = image
                counts[label_id] += 1

    return test_images


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
