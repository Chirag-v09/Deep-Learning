# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:13:14 2020

@author: Chirag
"""

from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing

max_features = 10000
maxlen = 20

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features)

X_train = preprocessing.sequence.pad_sequences(X_train, maxlen = maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen = maxlen)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length = maxlen))

model.add(Flatten())

model.add(Dense(1, activation = 'sigmoid'))

model.compile('rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)

