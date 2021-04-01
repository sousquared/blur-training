#!pip install robustness==1.1  # (or 1.1.post2)

import os
import sys

sys.path.append("../../")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

from training.utils import GaussianBlurAll, AverageMeter, accuracy

cudnn.benchmark = True

arch = sys.argv[1]
epoch = 60
MODELS_DIR = "/mnt/work/blur-training/imagenet16/logs/models/"
RESULTS_DIR = "./results/{}/".format(arch)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# random seed settings
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)


def load_data(
    batch_size, in_path="/mnt/data/ImageNet/ILSVRC2012/", in_info_path="../../info/"
):
    """
    load 16-class-ImageNet
    :param batch_size: the batch size used in training and test
    :param in_path: the path to ImageNet
    :param in_info_path: the path to the directory
                              that contains imagenet_class_index.json, wordnet.is_a.txt, words.txt
    :return: train_loader, test_loader
    """

    in_hier = ImageNetHierarchy(in_path, in_info_path)
    superclass_wnid = common_superclass_wnid("geirhos_16")  # 16-class-imagenet
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

    ### parameters for normalization: choose one of them if you want to use normalization #############
    # normalize = transforms.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    # https://github.com/MadryLab/robustness/blob/master/robustness/datasets.py

    # If you want to use normalization parameters of ImageNet from pyrotch:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # 16-class-imagenet
    # normalize = transforms.Normalize(mean=[0.4677, 0.4377, 0.3986], std=[0.2769, 0.2724, 0.2821])
    # normalize = transforms.Normalize(mean=[0.4759, 0.4459, 0.4066], std=[0.2768, 0.2723, 0.2827])
    ############################################################################
    # add normalization
    custom_dataset.transform_train.transforms.append(normalize)
    custom_dataset.transform_test.transforms.append(normalize)

    # train_loader, test_loader = custom_dataset.make_loaders(workers=10,
    #                                                         batch_size=batch_size)
    train_loader, test_loader = custom_dataset.make_loaders(
        workers=10, batch_size=batch_size, only_val=True
    )

    return train_loader, test_loader


def load_model(model_path, arch, num_classes=16):
    """
    :param model_path: path to the pytorch saved file of the model you want to use
    """
    model = models.__dict__[arch]()
    # change the number of last layer's units
    model = models.__dict__[arch]()
    if (
        arch.startswith("alexnet")
        or arch.startswith("vgg")
        or arch.startswith("mnasnet")
        or arch.startswith("mobilenet")
    ):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif (
        arch.startswith("resne")
        or arch.startswith("shufflenet")
        or arch.startswith("inception")
        or arch.startswith("wide_resnet")
    ):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif arch.startswith("squeezenet"):
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels,
            num_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    checkpoint = torch.load(model_path, map_location="cuda:0")
    model.load_state_dict(checkpoint["state_dict"])

    return model


def calc_band_acc(model, sigma1=0, sigma2=0, raw=False):
    """
    Args:
        model: model to test
        sigma1, sigma2: bandpass images are made by subtracting
            GaussianBlur(sigma1) - GaussianBlur(sigma2)
        raw: if True, calculate accuracy of raw images
    return: accuracy of bandpass images
        :: when raw == True, return accuracy of raw images
    """
    global test_loader
    global device

    top1 = AverageMeter("Acc@1", ":6.2f")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0], data[1].to(device)
            if not raw:
                if sigma1 == 0:
                    low1 = inputs  # raw images
                else:
                    low1 = GaussianBlurAll(inputs, sigma1)  # 1st lowpass images
                low2 = GaussianBlurAll(inputs, sigma2)  # 2nd lowpass images
                inputs = low1 - low2  # bandpass images
            inputs = inputs.to(device)
            outputs = model(inputs)
            acc1 = accuracy(outputs, labels, topk=(1,))
            top1.update(acc1[0], inputs.size(0))

    return top1.avg


def test_performance(model_name, epoch=60):
    """
    compute performance of the model
    and return the results as lists
    """
    # set paths
    model_path = os.path.join(MODELS_DIR, model_name, "epoch_{}.pth.tar".format(epoch))
    save_path = os.path.join(
        RESULTS_DIR, "{}_e{}_acc1.csv".format(model_name, epoch)
    )  # save path of the result

    # load model
    model = load_model(model_path, arch).to(device)

    acc1_list = []
    acc1 = calc_band_acc(model, sigma1=0, sigma2=1)  # raw - s1
    acc1_list.append(acc1.item())
    sigma_list = [2 ** i for i in range(6)]
    for i in sigma_list:
        acc1 = calc_band_acc(model, sigma1=i, sigma2=i * 2)
        acc1_list.append(acc1.item())
    acc1 = calc_band_acc(model, raw=True)
    acc1_list.insert(0, acc1.item())

    # range of sigma
    s = ["s{}-s{}".format(i, i * 2) for i in sigma_list]
    s.insert(0, "raw-s1")
    s.insert(0, "raw")

    # make dataframe and save
    df = pd.DataFrame(np.array(acc1_list).reshape(1, -1), index=[model_name], columns=s)
    df.to_csv(save_path)


# test

# data settings
batch_size = 64

_, test_loader = load_data(batch_size)

# models to compare
modes = [
    "normal",
    "all",
    "mix",
    "random-mix",
    "single-step",
    "fixed-single-step",
    "reversed-single-step",
    "multi-steps",
]

# sigmas to compare
sigmas_mix = [s for s in range(1, 6)] + [10]
sigmas_random_mix = ["00-05", "00-10"]

# make model name list
model_names = []
for mode in modes:
    if mode in ("normal", "multi-steps"):
        model_names += [f"{arch}_{mode}"]
    elif mode == "random-mix":
        for min_max in sigmas_random_mix:
            model_names += [f"{arch}_{mode}_s{min_max}"]
    elif mode == "mix":
        for sigma in sigmas_mix:
            model_names += [f"{arch}_{mode}_s{sigma:02d}"]
    else:
        for s in range(4):
            model_names += [f"{arch}_{mode}_s{s + 1:02d}"]

for model_name in model_names:
    test_performance(model_name, epoch)
