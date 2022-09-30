# Internal imports
from typing import Callable

# External imports
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Custom import
from adaptive_neighbourhoods.distances import inverse_multiquadric, inverse_quadric
from adaptive_neighbourhoods._types import Arr, Number


def _is_touching(x, y, r, dist, indexes):
    n_samples = x.shape[0]
    touching = np.zeros_like(y)
    pairwise_radius = r[...,None]+r[None,...]

    for i in range(0, n_samples):
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
        iterative: bool = False) -> np.ndarray:
    """Create adapted and unique neighbourhoods.

    Considering the density information and class distribution of `x`
    and `y`, generate the upper bound of the neighbourhoods to search
    for adversarial examples.

    Parameters
    ----------
    x : np.ndarray
        The input data to generate neighbourhoods around, these will
        generally be your training data for which adversarial
        neighbourhoods will be searched within.
    y : np.ndarray
        Encoded repreentations of classes for each data point in
        `x`. For example, if we have a two class problem with 4 data
        points, `y` could look like np.array([0, 1, 1, 0]).
    step_size : Number
        The initial step size for expanding the neighbourhoods. This
        number should be suitably small so as to not overlap with any
        other points from different classes before being modulated by
        the density information.
    distance_fn : Callable[[Arr, Arr], Arr]
        A callable metric function to determine the distance between
        neighbouring data points from which estimated density is
        computed. By default, we use the inverse multiquadric
        function.
    iterative : bool
        Boolean parameter that, if True, will return neighbourhood
        size from each successive iteration of the algorithm that
        allows you to visualise the expansion of the neighbourhoods.

    Returns
    -------
    np.ndarray
        A numpy array of neighbourhood sizes for each data point in
        `x`. If iterative is True, then the return will contain the
        neighbourhood sizes for each iteration of this algorithm.

    Examples
    --------
    FIXME: Add docs.

    """
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
    for i in range(0, n_samples):
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
        t = _is_touching(x, y, radii, dist, diff_class_indexes)
        t[np.where(_step_size < 1e-20)] = 1
        not_touching = np.where(t != 1)[0]
        radii[not_touching] += _step_size[not_touching]
        n_steps += 1
        _step_size *= d

    if not iterative:
        return radii
    else:
        return np.array(radiis)
