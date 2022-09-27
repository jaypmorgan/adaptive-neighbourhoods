# Internal imports
from typing import Callable

# External imports
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Custom import
from adaptive_neighbourhoods.distances import inverse_multiquadric, inverse_quadric
from adaptive_neighbourhoods._types import Arr, Number


def is_touching(x, y, r, dist, indexes):
    n_samples = x.shape[0]
    touching = np.zeros_like(y)
    pairwise_radius = r[...,None]+r[None,...]

    for i in range(n_samples):
        for j in indexes[y[i]]:
            if i != j and not (touching[i] or touching[j]):
                if dist[i,j] <= r[i]+r[j]:
                    touching[i] = 1
                    touching[j] = 1
    return touching


def epsilon_expand(
        x: np.ndarray,
        y: np.ndarray,
        step_size: Number = 1e-7,
        distance_fn: Callable[[Arr, Arr], Arr] = inverse_quadric,
        iterative: bool = False):
    n_samples   = x.shape[0]
    t           = np.zeros((n_samples,)) # are neighbourhoods overlapping
    n_steps     = 0                                 # number of steps for debugging

    classes = np.unique(y)
    same_class_indexes = {k: np.where(y == k)[0] for k in classes}
    diff_class_indexes = {k: np.where(y != k)[0] for k in classes}

    dist = squareform(pdist(x))
    _inv = 1 / np.sqrt(1 + (2 * dist)**2)
    np.fill_diagonal(_inv, 1.0)

    d = np.zeros_like(t)
    for i in range(n_samples):
        c = same_class_indexes[y[i]]
        d[i] = np.sum(_inv[i, c]) / c.shape[0]
    #d = (1 - d)

    np.fill_diagonal(dist, np.inf)

    radii = np.zeros_like(y, float)
    #radii = dist.min(0)/2
    #_step_size  = radii/2
    _step_size=np.repeat([step_size], n_samples)
    radiis = []

    while not np.all(t == 1):
        if iterative:
            radiis.append(radii.copy())
        t = is_touching(x, y, radii, dist, diff_class_indexes)
        t[np.where(_step_size < 1e-20)] = 1
        not_touching = np.where(t != 1)[0]
        radii[not_touching] += _step_size[not_touching]
        n_steps += 1
        _step_size *= d

    if not iterative:
        return radii
    else:
        return radiis
