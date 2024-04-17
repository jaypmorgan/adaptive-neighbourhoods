import pandas as pd
import numpy as np
import adaptive_neighbourhoods as adpt

def iris():

    iris = pd.read_csv("tests/data/iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    mapper = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    color_mapper = {0: 'red', 1: 'green', 2: 'blue'}
    
    x = iris.drop(columns='species').to_numpy()
    y = np.array([mapper[i] for i in iris['species'].tolist()])

    same_class_indexes = []
    for i in range(3):
        same_class_indexes.append([j for j in range(150) if y[j] != i])

    density = adpt.compute_density(x, y, 5, same_class_indexes)
    radius = adpt.epsilon_expand(x[:, 0:2], y, max_step_size=0.05, show_progress=False)
    return radius


def test_iris():
    radius = iris()
    assert len(radius) == 150
    assert np.isclose(radius[0], 0.1563875)
    assert radius[-1] == 0.0

