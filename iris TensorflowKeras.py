
import tensorflow as tf

from sklearn.datasets import load_iris
dataset = load_iris()

X = dataset.data
y = dataset.target

from tensorflow import keras
from tensorflow.keras import losses

model = keras.models.Sequential()
model.add(keras.layers.Dense(128, activation = "tanh", input_shape = (4, )))
model.add(keras.layers.Dense(32, activation = "relu"))
model.add(keras.layers.Dense(3, activation = 'softmax'))

model.compile("rmsprop", loss = losses.sparse_categorical_crossentropy, metrics = ["accuracy"])

model.fit(X, y, epochs = 5)

model.summary()

