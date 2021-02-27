import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

arch = "alexnet"
epoch = 60

# directories and model settings
DATA_DIR = f"./results/{arch}"
OUTPUTS_DIR = f"./plots/{arch}"
if not os.path.exists(OUTPUTS_DIR):
    os.makedirs(OUTPUTS_DIR)


def read_result(model_name, epoch=60, value="acc1"):
    file_path = os.path.join(DATA_DIR, "{}_e{}_{}.csv".format(model_name, epoch, value))
    return pd.read_csv(file_path, index_col=0)


'''
def read_raw_acc1(model_name, epoch):
    """Return top-1 accuracy from saved model"""
    model_path = os.path.join(MODELS_DIR, model_name, 'epoch_{}.pth.tar'.format(epoch))
    checkpoint = torch.load(model_path, map_location='cpu')

    return checkpoint['val_acc'].item()
'''

# read  normal results
mode = "normal"
model_name = "{}_{}".format(arch, mode)
normal_acc1 = read_result(model_name, epoch).values[0]

color_list = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot(mode):
    # read band-pass accuracy results
    acc1 = []
    if mode == "multi-steps":
        model_name = "{}_{}".format(arch, mode)
        acc1 = read_result(model_name, epoch).values[0]
    else:
        for i in range(1, 5):
            model_name = "{}_{}_s{}".format(arch, mode, i)
            acc1.append(read_result(model_name, epoch).values[0])

    x = ["σ{}-σ{}".format(2 ** i, 2 ** (i + 1)) for i in range(6)]
    x.insert(0, "raw - σ1")
    x.insert(0, "raw")

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(
        1,
        1,
        1,
        title="Top-1 Accuracy of Band-Pass Images, {} {}".format(
            arch.capitalize(), mode.capitalize()
        ),
        xlabel="Test images",
        ylabel="Top-1 accuracy",
        ylim=(0, 1),
    )
    ax.plot(x[0], normal_acc1[0], marker="o", color=color_list[0])
    ax.plot(x[1:], normal_acc1[1:], label="normal", marker="o")
    if mode == "multi-steps":
        ax.plot(x[0], acc1[0], marker="o", c="mediumvioletred")
        ax.plot(x[1:], acc1[1:], label=mode, marker="o", c="mediumvioletred")
    else:
        for i in range(4):
            ax.plot(x[0], acc1[i][0], marker="o", color=color_list[i + 1])
            ax.plot(x[1:], acc1[i][1:], label="σ={}".format(i + 1), marker="o")
    ax.legend()
    # ax.set_xticks(np.arange(0, max_sigma+1, 5))
    plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(10))
    # ax.xaxis.set_major_locator(tick.MultipleLocator(1))
    ax.grid(which="major")
    ax.grid(which="minor")
    # plt.xlim()
    plt.ylim(0, 100)
    fig.show()
    filename = "bandpass-acc1_{}_{}_e{}.png".format(arch, mode, epoch)
    fig.savefig(os.path.join(OUTPUTS_DIR, filename))


mode = "all"
plot(mode)

mode = "mix"
plot(mode)

mode = "single-step"
plot(mode)

mode = "reversed-single-step"
plot(mode)

mode = "multi-steps"
plot(mode)