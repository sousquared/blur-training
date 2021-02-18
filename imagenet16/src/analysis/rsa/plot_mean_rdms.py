import os
import pathlib
import pickle
import sys

import seaborn as sns
from matplotlib import pyplot as plt

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(str(current_dir) + "/../../../")

from src.analysis.rsa.rdm import alexnet_layers
from src.dataset.imagenet16 import label_map


def load_rdms(in_dir, model_name, epoch):
    file_path = os.path.join(in_dir, model_name + f"_e{epoch:02d}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_rdms(in_dir, model_name, epoch, out_dir):
    """Plot several layers RDMs in one figure.
    Args:
        model_name: name of the model to examine.
    """
    mean_rdms = load_rdms(in_dir=in_dir, model_name=model_name, epoch=epoch)

    # get analysis parameters.
    num_images = mean_rdms["num_images"]
    num_filters = mean_rdms["num_filters"]
    target_id = mean_rdms["target_id"]

    fig = plt.figure(dpi=300)
    for i, layer in enumerate(alexnet_layers):
        ax = fig.add_subplot(2, 4, i + 1)
        # sns.set(font_scale=0.5)  # adjust the font size of labels
        ax.set_title(layer)

        sns.heatmap(
            mean_rdms[layer],
            ax=ax,
            square=True,
            vmin=0,
            vmax=2,
            xticklabels=False,
            yticklabels=False,
            cmap="coolwarm",
            cbar=False,
            cbar_ax=None,
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
    #     cbar_ax = fig.add_axes([.91, .3, .03, .4])
    #     sns.heatmap(rdm, cbar=True, cbar_ax=cbar_ax,
    #                 vmin=0, vmax=2, cmap='coolwarm',
    #                 xticklabels=False, yticklabels=False)

    title = "{}, {}, epoch={}".format(
        *[n.capitalize() for n in model_name.split("_", 1)], epoch
    )
    #     sns.set(font_scale=0.5)  # adjust the font size of title
    fig.suptitle(title)
    # fig.tight_layout(rect=[0, 0, .9, 1])
    fig.tight_layout()
    filename = "rdms_{}_e{}_{}_n{}.png".format(
        model_name, epoch, label_map[target_id], num_images
    )
    plt.savefig(os.path.join(out_dir, filename))
    # plt.show()


if __name__ == "__main__":
    arch = "alexnet"
    epoch = 60
    analysis_name = f"{arch}_bandpass"
    in_dir = f"./results/{analysis_name}"
    out_dir = f"./plots/{analysis_name}"

    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # plot settings
    # models to plot
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

    # plot and save
    for mode in modes:
        if mode in ("normal", "multi-steps"):
            model_name = f"{arch}_{mode}"
            plot_rdms(
                in_dir=in_dir,
                model_name=model_name,
                epoch=epoch,
                out_dir=out_dir,
            )
        elif mode == "random-mix":
            for min_max in sigmas_random_mix:
                model_name = "{}_{}_s{}".format(arch, mode, min_max)
                plot_rdms(
                    in_dir=in_dir,
                    model_name=model_name,
                    epoch=epoch,
                    out_dir=out_dir,
                )
        elif mode == "mix":
            for sigma in sigmas_mix:
                model_name = f"{arch}_{mode}_s{sigma:02d}"
                plot_rdms(
                    in_dir=in_dir,
                    model_name=model_name,
                    epoch=epoch,
                    out_dir=out_dir,
                )
        else:
            for s in range(4):
                model_name = f"{arch}_{mode}_s{s + 1:02d}"
                plot_rdms(
                    in_dir=in_dir,
                    model_name=model_name,
                    epoch=epoch,
                    out_dir=out_dir,
                )
