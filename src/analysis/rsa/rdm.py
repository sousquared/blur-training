import os
import pathlib
import pickle
import sys

import seaborn as sns
from matplotlib import pyplot as plt

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../"))

from src.analysis.rsa.rsa import alexnet_layers

# from src.dataset.imagenet16 import label_map


def load_rdms(in_dir, model_name, epoch):
    file_path = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def plot_rdms(rdms, vmin=0, vmax=2, title="", out_file="rdms.png", show_plot=False):
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
        # "all",
        # "mix",
        # "random-mix",
        # "single-step",
        # "fixed-single-step",
        # "reversed-single-step",
        # "multi-steps",
    ]

    # sigmas to plot
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

    # load, plot and save
    for model_name in model_names:
        # load rdms
        rdms = load_rdms(in_dir=in_dir, model_name=model_name, epoch=epoch)

        # get analysis parameters.
        num_images = rdms["num_images"]
        num_filters = rdms["num_filters"]

        # (optional) set title of the plot
        title = f"{arch}, {model_name}, epoch={epoch}"

        # set the filename
        filename = "mean-rdms_{}_e{}_f{}_n{}.png".format(
            model_name, epoch, num_filters, num_images
        )  # add "target_id" if you need it.
        out_file = os.path.join(out_dir, filename)

        plot_rdms(
            rdms=rdms, vmin=0, vmax=2, title=title, out_file=out_file, show_plot=False
        )
