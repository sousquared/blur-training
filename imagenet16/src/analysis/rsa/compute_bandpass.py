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
from src.dataset.imagenet16 import label_map


def analyze(
    models_dir: str,
    arch: str,
    model_name: str,
    epoch: int,
    test_images: torch.Tensor,
    target_id: int,
    num_filters: int,
    num_images: int,
    out_dir: str,
):
    model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
    model = load_model(arch=arch, model_path=model_path).to(device)
    mean_rdms = compute_mean_rdms(model=model, test_images=test_images)
    # add parameter settings of this analysis
    mean_rdms["target_id"] = target_id
    mean_rdms["num_filters"] = num_filters
    mean_rdms["num_images"] = num_images
    save_mean_rdms(
        mean_rdms=mean_rdms, out_dir=out_dir, model_name=model_name, epoch=epoch
    )


def compute_mean_rdms(model, test_images) -> dict:
    """Computes and save mean RDMs.
    Args:
        test_images: images to test the model with. shape=(N, F+1, C, H, W)
            Where: F is the number of filters.
                F+1 means filter applied images(F) and a raw image(+1)
    """
    RDM = AlexNetRDM(model)
    mean_rdms = RDM.compute_mean_rdms(test_images)

    return mean_rdms


def save_mean_rdms(mean_rdms: dict, out_dir: str, model_name: str, epoch: int):
    # save dict object
    file_name = model_name + f"_e{epoch:02d}.pkl"
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(mean_rdms, f)


if __name__ == "__main__":
    """Run example for normal alexnet with band-pass test images."""
    # arguments
    arch = "alexnet"
    mode = "normal"
    model_name = f"{arch}_{mode}"
    epoch = 60

    # I/O settings
    models_dir = "/mnt/work/blur-training/imagenet16/logs/models/"  # model directory
    out_dir = f"./results/{arch}_bandpass"
    test_images_dir = "./test-images"  # directory for test images overview file
    assert os.path.exists(models_dir), f"{models_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)

    # data settings
    # num_data = 1600
    target_id = 1  # bear
    num_filters = 6  # number of band-pass filters
    num_images = 10  # number of images for each class.

    # random seed settings
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load and make test images
    test_images = make_bandpass_images(
        target_id=target_id, num_filters=num_filters, num_images=num_images
    ).to(device)

    # change the order of num_images and num_filters(+1)
    test_images = test_images.transpose(1, 0)

    # save test images
    image_name = f"bandpass_{label_map[target_id]}_f{num_filters}_n{num_images}.png"
    imsave(
        torchvision.utils.make_grid(
            test_images.reshape(-1, *test_images.shape[2:]).cpu(),
            nrow=num_filters + 1,
        ),
        filename=os.path.join(test_images_dir, image_name),
        unnormalize=True,
    )

    analyze(
        models_dir=models_dir,
        arch=arch,
        model_name=model_name,
        epoch=epoch,
        test_images=test_images,
        target_id=target_id,
        num_filters=num_filters,
        num_images=num_images,
        out_dir=out_dir,
    )
    # # load trained model
    # model_path = os.path.join(models_dir, model_name, f"epoch_{epoch:02d}.pth.tar")
    # model = load_model(arch=arch, model_path=model_path).to(device)
    #
    # mean_rdms = compute_mean_rdms(model=model, test_images=test_images)
    #
    # # add parameter settings of this analysis
    # mean_rdms["target_id"] = target_id
    # mean_rdms["num_filters"] = num_filters
    # mean_rdms["num_images"] = num_images
    #
    # save_mean_rdms(
    #     mean_rdms=mean_rdms, out_dir=out_dir, model_name=model_name, epoch=epoch
    # )