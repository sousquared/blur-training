import os
import pathlib
import sys

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.activations import load_activations
from src.analysis.rsa.rdm import save_rdms


def compute_mean_rdms(
    in_dir: str,
    num_filters: int = 6,
    num_images: int = 1600,
) -> dict:
    """Computes RDM for each image and return mean RDMs.
    Args:
        in_dir: path to input directory
        num_filters: number of band-pass filter
        num_images: number of images
    Returns: Mean RDMs (Dict)
    """
    mean_rdms = {}
    mean_rdms["num_filters"] = num_filters
    mean_rdms["num_images"] = num_images
    # mean_rdms["target_id"] = target_id

    for layer in alexnet_layers:
        rdms = []
        # compute RDM for each image (with some filters applied)
        for image_id in range(num_images):
            file_name = f"image{image_id:04d}_f{num_filters:02d}.pkl"
            activations = load_activations(in_dir=in_dir, file_name=file_name)
            activation = activations[layer].reshape(num_filters + 1, -1)
            rdm = squareform(pdist(activation, metric="correlation"))
            rdms.append(rdm)

        rdms = np.array(rdms)

        mean_rdms[layer] = rdms.mean(0)

    return mean_rdms


def plot_bandpass_rdms(
    rdms, num_filters, vmin=0, vmax=2, title="", out_file="rdms.png", show_plot=False
):
    """Plot several layers RDMs in one figure.
    Args:
        model_name: name of the model to examine.
    """
    fig = plt.figure(dpi=300)

    for i, layer in enumerate(alexnet_layers):
        ax = fig.add_subplot(2, 4, i + 1)
        # sns.set(font_scale=0.5)  # adjust the font size of labels
        ax.set_title(layer)

        sns.heatmap(
            rdms[layer],
            ax=ax,
            square=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            yticklabels=["0", "0-1", "1-2", "2-4", "4-8", "8-16", "16-"],
            cmap="coolwarm",
            cbar=False,
            # cbar_ax=cbar_ax,
        )

        ax.hlines(
            [i for i in range(1, num_filters + 1)],
            *ax.get_xlim(),
            linewidth=0.1,
            colors="gray",
        )
        ax.vlines(
            [i for i in range(1, num_filters + 1)],
            *ax.get_ylim(),
            linewidth=0.1,
            colors="gray",
        )

    # show color bar
    # cbar_ax = fig.add_axes([.91, .3, .03, .4])
    # sns.heatmap(rdms[layer], cbar=True, cbar_ax=cbar_ax,
    #             vmin=-1, vmax=1, cmap='coolwarm',
    #             xticklabels=False, yticklabels=False)

    # sns.set(font_scale=0.5)  # adjust the font size of title
    if title:
        fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.tight_layout()
    plt.savefig(out_file)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    arch = "alexnet"
    mode = "normal"
    model_name = f"{arch}_{mode}"
    epoch = 60

    # I/O settings
    data_dir = "./results/activations/alexnet/"
    results_dir = f"./results/mean_rdms/{arch}/"
    plots_dir = f"./plots/mean_rdms/{arch}/"

    assert os.path.exists(data_dir), f"{data_dir} does not exist."
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    in_dir = os.path.join(data_dir, f"{model_name}_e{epoch:02d}")
    assert os.path.exists(in_dir), f"{in_dir} does not exist."

    mean_rdms = compute_mean_rdms(in_dir=in_dir, num_filters=6, num_images=1600)

    save_rdms(
        mean_rdms=mean_rdms, out_dir=results_dir, model_name=model_name, epoch=epoch
    )

    # get analysis parameters.
    num_images = mean_rdms["num_images"]
    num_filters = mean_rdms["num_filters"]

    # (optional) set title of the plot
    title = f"{arch}, {model_name}, epoch={epoch}"

    # set the filename
    filename = "mean-rdms_{}_e{}_f{}_n{}.png".format(
        model_name, epoch, num_filters, num_images
    )  # add "target_id" if you need it.
    plot_file = os.path.join(plots_dir, filename)

    plot_bandpass_rdms(
        rdms=mean_rdms,
        num_filters=num_filters,
        vmin=0,
        vmax=2,
        title=title,
        out_file=plot_file,
        show_plot=False,
    )
