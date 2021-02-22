import os
import pathlib

import torchvision.transforms as transforms
from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

current_dir = pathlib.Path(os.path.abspath(__file__)).parent

num_classes = 16
num_channels = 3
height = 224
width = 224

categories = sorted(
    [
        "knife",
        "keyboard",
        "elephant",
        "bicycle",
        "airplane",
        "clock",
        "oven",
        "chair",
        "bear",
        "boat",
        "cat",
        "bottle",
        "truck",
        "car",
        "bird",
        "dog",
    ]
)
label_map = {k: v for k, v in enumerate(categories)}


def load_data(
    batch_size: int,
    in_path: str = "/mnt/data/ImageNet/ILSVRC2012/",
    in_info_path: str = str(current_dir) + "/info/",
    normalize: bool = True,
):
    """
    load 16-class-ImageNet
    Arguments:
        batch_size (int): the batch size used in training and test
        in_path (str): the path to ImageNet
        in_info_path (str): the path to the directory that contains
                    imagenet_class_index.json, wordnet.is_a.txt, words.txt
        normalize (bool): Use normalization on images or nor. (default: True)
    Returns: train_loader, test_loader
    """

    # 16-class-imagenet
    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid("geirhos_16")
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

    if normalize:
        # standard ImageNet normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # add normalization
        custom_dataset.transform_train.transforms.append(normalize)
        custom_dataset.transform_test.transforms.append(normalize)

    train_loader, test_loader = custom_dataset.make_loaders(
        workers=10, batch_size=batch_size
    )

    return train_loader, test_loader
