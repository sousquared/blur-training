def adjust_learning_rate(optimizer, epoch, args, every=20):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def adjust_multi_steps(epoch: int):
    """For 'multi-steps' mode.
        Return sigma based on current epoch of training.
    Args:
        epoch: current epoch of training
    Returns: sigma
    """
    if epoch < 10:
        sigma = 4
    elif epoch < 20:
        sigma = 3
    elif epoch < 30:
        sigma = 2
    elif epoch < 40:
        sigma = 1
    else:
        sigma = 0  # no blur

    return sigma


def adjust_multi_steps_cbt(init_sigma, epoch, decay_rate=0.9, every=5):
    """
    Sets the sigma of Gaussian Blur decayed every 5 epoch.
    This is for 'multi-steps-cbt' mode.
    This idea is based on "Curriculum By Texture"
    Args:
        init_sigma: initial sigma of Gaussian kernel
        epoch: training epoch at the moment
        decay_rate: how much the model decreases the sigma value
        every: the number of epochs the model decrease sigma value
    Return: sigma of Gaussian kernel (GaussianBlur)
    """
    return init_sigma * (decay_rate ** (epoch // every))
