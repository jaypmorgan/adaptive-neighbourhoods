import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import adaptive_neighbourhoods as adpt

iris = pd.read_csv("./iris.data", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
mapper = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
color_mapper = {0: 'red', 1: 'green', 2: 'blue'}

x = iris.drop(columns='species').to_numpy()
y = np.array([mapper[i] for i in iris['species'].tolist()])

same_class_indexes = []
for i in range(3):
    same_class_indexes.append([j for j in range(150) if y[j] == i])

density = adpt.compute_density(x, y, 5, same_class_indexes)
radius = adpt.epsilon_expand(x[:, 0:2], y, max_step_size=0.05, show_progress=True)

circles = []
for xi, yi, r in zip(x[:, 0], x[:, 1], radius):
    circles.append(plt.Circle((xi, yi), r, color='black', linewidth=0.2, fill=False))
circles_2 = []
for xi, yi, r in zip(x[:, 0], x[:, 1], density):
    circles_2.append(plt.Circle((xi, yi), r, color='black', linewidth=0.2, fill=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
c = [color_mapper[i] for i in y]
ax[1].scatter(x[:, 0], x[:, 1], c=c)
ax[0].scatter(x[:, 0], x[:, 1], c=c)
for i in range(len(circles)):
    ax[1].add_patch(circles[i])
    ax[0].add_patch(circles_2[i])

ax[0].set_title("Density")
ax[1].set_title("Adapted Neighbourhoods")
