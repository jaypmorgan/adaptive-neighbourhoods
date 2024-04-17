# external imports
import dfp
import torch
import torchvision
import numpy as np

# custom imports
import adaptive_neighbourhoods as adpt


# download the mnist dataset
loader = lambda is_train: torchvision.datasets.MNIST("data/mnist", train=is_train, download=True)
mnist_tr = loader(is_train=True)
mnist_te = loader(is_train=False)

# create adapted neighbourhoods for each point in the training set
radius = adpt.epsilon_expand(
    (mnist_tr.data.reshape(-1, 28*28)/255.0).numpy(),
    mnist_tr.targets.numpy(),
    max_step_size=0.5, show_progress=True)

dfp.port_pickle("data/mnist_radius.pkl", radius)
