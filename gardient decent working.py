
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

m = 100
X = np.random.randn(m,1)
y = 5*X+7+np.random.randn(m,1)

plt.scatter(X, y)
plt.show()

n_iter = 1000
eta = 0.1
theta = np.random.randn(2, 1)

X = np.c_[X, np.ones(m)]

for i in range(n_iter):
    gradients = 2/m*X.T@((X @ theta) - y)
    theta = theta - eta*gradients

# Here theta is wieghts that updated through the gradient decent
