import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaptive_neighbourhoods import epsilon_expand

iris = pd.read_csv("iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
mapper = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
color_mapper = {0: 'red', 1: 'green', 2: 'blue'}

x = iris.drop(columns='species').to_numpy()
y = np.array([mapper[i] for i in iris['species'].tolist()])
c = [color_mapper[i] for i in y]

radius = epsilon_expand(x, y, step_size=0.01)

circles = []
for xi, yi, r in zip(x[:, 0], x[:, 1], radius):
    circles.append(
        plt.Circle((xi, yi), r, color='black', fill=False)
    )

fig, ax = plt.subplots()
ax.scatter(x[:, 0], x[:, 1], c=c)
for circle in circles:
    ax.add_patch(circle)