import matplotlib.pyplot as plt
import torch


def imshow(img, title="", filename="", scale=True):
    # img = img / 4 + 0.5     # unnormalize
    if type(img) == torch.Tensor:
        img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)
    if not scale:
        plt.xticks([])  # if you want to remove scale axes
        plt.yticks([])
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)  # added for saving the image
    plt.show()
