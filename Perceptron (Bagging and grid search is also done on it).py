import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import Perceptron
pn = Perceptron()
pn.fit(X_train, y_train)
print(pn.score(X_test, y_test))
print(pn.score(X_train, y_train))

''' Bagging '''
from sklearn.ensemble import BaggingClassifier
bgp = BaggingClassifier(pn, n_estimators = 5)
bgp.fit(X_train, y_train)
print(bgp.score(X_test, y_test))


''' Grid Search '''
param_grid = [{'max_iter' : [1000, 5000, 10000]},
             {'eta0' : [1.0, 2.0, 3.0]},

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pn, param_grid)
grid.fit(X_train, y_train)

print("Best Estimator:- ", grid.best_estimator_)
print("Best Index:- ", grid.best_index_)
print("Best Parameter;- ", grid.best_params_)
print("Best Score:- ", grid.best_score_)