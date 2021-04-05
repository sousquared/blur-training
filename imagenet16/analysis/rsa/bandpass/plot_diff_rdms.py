import os
import pathlib
import sys

# add the path to load src module
current_dir = pathlib.Path(os.path.abspath(__file__)).parent
sys.path.append(os.path.join(str(current_dir), "../../../../"))

from src.analysis.rsa.rsa import alexnet_layers
from src.analysis.rsa.rdm import load_rdms
from src.analysis.rsa.bandpass.mean_rdms import plot_bandpass_rdms

if __name__ == "__main__":
    in_dir = "./results/mean_rdms/alexnet/"
    out_dir = "./plots/diff_rdms/"
    assert os.path.exists(in_dir), f"{in_dir} does not exist."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    arch = "alexnet"
    epoch = 60

    mode = "normal"
    model_name = f"{arch}_{mode}"
    rdms_normal = load_rdms(in_dir=in_dir, model_name=model_name, epoch=epoch)

    mode = "mix_s04"
    model_name = f"{arch}_{mode}"
    rdms_mix04 = load_rdms(in_dir=in_dir, model_name=model_name, epoch=epoch)

    diff_rdms = {}
    for layer in alexnet_layers:
        diff_rdms[layer] = rdms_mix04[layer] - rdms_normal[layer]

    # get analysis parameters.
    num_images = rdms_normal["num_images"]
    num_filters = rdms_normal["num_filters"]

    # (optional) set title
    title = f"{arch}, mix_s04 - normal, epoch={epoch}"

    # set filename
    filename = "diff_normal_mix04_e{}_f{}_n{}.png".format(
        epoch, num_filters, num_images
    )  # add "target_id" if you need it.
    out_file = os.path.join(out_dir, filename)

    # plot_rdms(rdms=diff_rdms, out_file=out_file, plot_show=True)
    plot_bandpass_rdms(
        rdms=diff_rdms,
        num_filters=num_filters,
        vmin=-0.5,
        vmax=0.5,
        title=title,
        out_file=out_file,
        show_plot=False,
    )