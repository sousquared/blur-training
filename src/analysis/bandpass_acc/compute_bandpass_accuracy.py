#!pip install robustness==1.1  # (or 1.1.post2)

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.image_process.lowpass_filter import GaussianBlurAll
from src.utils.accuracy import AverageMeter, accuracy
from src.utils.model import load_model
from src.dataset.imagenet16 import load_data

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
