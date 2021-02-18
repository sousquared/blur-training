import os
import pathlib
import pickle
import sys

import numpy as np
import torch
import torchvision

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.utils.model import load_model
from src.utils.image import imsave
from src.analysis.rsa.rdm import AlexNetRDM
from src.analysis.rsa.bandpass_images import make_bandpass_images


if __name__ == "__main__":
    """Run example for normal alexnet with band-pass test images."""
    # arguments
    arch = "alexnet"
    mode = "normal"
    model_name = f"{arch}_{mode}"
    epoch = 60

    # I/O settings
    models_dir = "/mnt/work/blur-training/imagenet16/logs/models/"  # model directory
    out_dir = "./results/{}_bandpass".format(arch)
    test_images_dir = "./test-images"  # directory for test images overview file
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)

    # data settings
    # num_data = 1600
    target_id = 1  # bear
    num_images = 10  # number of images for each class.

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load and make test images
    test_images = make_bandpass_images(target_id, num_images).to(device)

    # change the order of num_images and num_filters(+1)
    test_images = test_images.transpose(1, 0)

    # save test images
    image_name = f"bandpass_n{num_images}.png"
    imsave(
        torchvision.utils.make_grid(
            test_images.reshape(-1, *test_images.shape[2:]).cpu(),
            nrow=test_images.shape[1],
        ),
        filename=os.path.join(test_images_dir, image_name),
        unnormalize=True,
    )

    model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
    model = load_model(arch=arch, model_path=model_path).to(device)

    RDM = AlexNetRDM(model)
    mean_rdms = RDM.compute_mean_rdms(test_images)

    # save dict object
    file_path = os.path.join(out_dir, model_name + f"_e{epoch:02d}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(mean_rdms, f)
