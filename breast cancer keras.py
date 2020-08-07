# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:17:58 2019

@author: Chirag
"""


'''   BREAST CANCER   '''

from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape = (30, )))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(2, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 5, batch_size = 256)


''' 95% ACCURACY '''
