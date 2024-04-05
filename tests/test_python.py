import adaptive_neighbourhoods as adpt
import numpy as np

x = np.random.randn(10, 10)
y = np.random.randint(0, 1, 10)

r = adpt.epsilon_expand(x, y)
print(r, type(r), r.dtype)