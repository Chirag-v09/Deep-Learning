# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:39:52 2019

@author: Chirag
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wine_data.csv',
                      names = [
                               'index',
                               'alcohol',
                               'malic acid',
                               'ash',
                               'alkalinity_of_ash',
                               'magnesium',
                               'total_phenols',
                               'flavanoids',
                               'nonflavanoid_phenols',
                               'proanthocyanins',
                               'colour_intensity',
                               'hue',
                               'diluted_wines',
                               'proline'                               
                              ]                     
                      )

X = dataset.iloc[: , 1:14].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.datasets import load_wine
dataset = load_wine()
