# external imports
import torch
import torchvision

# custom imports
import adaptive_neighbourhoods as adpt


# download the mnist dataset
loader = lambda is_train: torchvision.datasets.MNIST("data/mnist", train=is_train, download=True)
mnist_tr = loader(is_train=True)
mnist_te = loader(is_train=False)

# create adapted neighbourhoods for each point in the training set
radius = adpt.epsilon_expand(mnist_tr.data, mnist_tr.targets, max_step_size=0.5, show_progress=True)
