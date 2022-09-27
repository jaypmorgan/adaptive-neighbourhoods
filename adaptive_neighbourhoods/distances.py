# Internal imports
from typing import Union

# External imports
import numpy as np
from einops import reduce

# Custom imports
from adaptive_neighbourhoods._types import Arr, Number


def norm(xi, xj):
    return xi-xj


def _kernel_denom(xi, xj, kernel_shape):
    return 1+kernel_shape * norm(xi, xj)**2


def gaussian(xi, xj, kernel_shape):
    return np.exp(-(kernel_shape * norm(xi, xj))**2)


def inverse_quadric(
        xi: Arr,
        xj: Arr,
        kernel_shape: Number = 5
) -> Arr:
    return 1 / _kernel_denom(xi, xj, kernel_shape)


def multiquadric(xi, xj, kernel_shape):
    return np.sqrt(_kernel_denom(xi, xj, kernel_shape))


def inverse_multiquadric(
        xi: Arr,
        xj: Arr,
        kernel_shape: Number = 5
) -> Arr:
    """Calculate the distance between $x_i$ and $x_j$ using the inverse multiquadric kernel.

    Parameters
    ----------
    xi : Arr
        Point 1
    xj : Arr
        Point 2
    kernel_shape : Number
        The width of the inverse multiquadric kernel. Higher values
        for this parameters makes the kernel width decrease.

    Returns
    -------
    Arr

    Examples
    --------
    FIXME: Add docs.


    """
    return 1 / np.sqrt(_kernel_denom(xi, xj, kernel_shape))
