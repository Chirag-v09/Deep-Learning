
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from keras.datasets import fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Scandal', 'Skirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize = (10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i] , cmap = matplotlib.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28, 28)))
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dense(132, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(train_images, train_label, epochs = 10)
c = model.evaluate(test_images, test_label)

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
