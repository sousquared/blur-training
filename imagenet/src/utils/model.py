import shutil

import torch
from torchvision import models


def save_checkpoint(state, is_best, param_path, epoch):
    filename = param_path + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, param_path + "model_best.pth.tar")


def save_model(state, param_path, epoch):
    filename = param_path + "epoch_{}.pth.tar".format(epoch)
    torch.save(state, filename)


def load_model(model_path, arch="alexnet"):
    """Load model.
    Args:
        model_path: Path to the pytorch saved file of the model you want to use
        arch: Architecture of CNN
    Returns: CNN model
    """
    checkpoint = torch.load(model_path, map_location="cuda:0")
    model = models.__dict__[arch]()
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        model.features = torch.nn.DataParallel(model.features)
        model.load_state_dict(checkpoint["state_dict"])
        # model.features = model.features.module  # if you want to disable Dataparallel
    return model
