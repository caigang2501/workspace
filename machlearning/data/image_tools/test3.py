import numpy as np
x = [-3, -2]
X, Y = np.meshgrid(x, np.linspace(-3, 3, 128))
print(X.shape,Y.shape)
