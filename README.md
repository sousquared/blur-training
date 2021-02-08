# blur-training
Training CNNs(Convolutional Neural Networks) with blurred images.


## Blurred images (GaussianBlur)
Training images are blurred by Gaussian function. The images are more blurred as std.(σ) of Gaussian kernel is bigger.
![blurred-images](./figures/blurred_images.png)


## Schedule
You can try different training schedule as for blurring images. Here is an overview of the training schedule: <br>
(The descriptions are different and some modes may not be included in `cifar10`)
![schedule](./figures/schedule.png)


# Datasets
Each repository contains the scripts for each dataset below:
- `imagenet/`: ImageNet dataset
- `imagenet16/`: 16-class-ImageNet dataset <br>
  This dataset is from Geirhos et al., 2018. <br>
  (R. Geirhos, C. R. M. Temme, J. Rauber, H. H. Schütt, M. Bethge and F. A. Wichmann: Generalisation in humans and deep neural networks. Advances in Neural Information Processing Systems (NeurIPS), 7538–7550, 2018.) <br>
  I made the dataset from ImageNet by using `robustness` library.
- `cifar10/`: Cifar-10 dataset  