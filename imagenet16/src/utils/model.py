import torch
import torch.nn as nn
import torchvision.models as models


def load_model(arch, num_classes=16):
    """
    Load model from pytorch model zoo and change the number of final layser's units
    Args:
        arch: name of architecture
        num_classes: number of last layer's units
    Returns: model (torch.model)
    """
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

    return model


def save_model(state, param_path, epoch):
    filename = param_path + "epoch_{}.pth.tar".format(epoch)
    torch.save(state, filename)