# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:05:43 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import imdb
(train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = 1000)

# As we restrict ourself to top 1000 frequent words so no words index will increase 999
max([max(sequence) for sequence in train_label])

# To read the review
words_index = imdb.get_word_index()
reverse_word_index = dict( \
    [(value, key) for (key, value) in words_index.items()])
decoded_review = ' '.join( \
    [reverse_word_index.get(i-3, '!') for i in train_label[0]])


def vectorize_sequences(sequences, dimensions = 1000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

y_train = np.asarray(train_label).astype('float32')
y_test = np.asarray(test_label).astype('float32')

X_val = X_train[:10000]
Partial_X_train = X_train[10000:]
y_val = y_train[:10000]
Partial_y_train = y_train[10000:]


from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import losses

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape = (1000, )))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop', loss = losses.binary_crossentropy, metrics = [metrics.binary_accuracy])
# optimizers.RMSprop(lr = 0.001)
# model.compile(optimizer = optimizers.RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

# model.compile(optimizer = optimizers.RMSprop(lr = 0.001), loss = losses.binary_crossentropy, metrics = [metrics.binary_accuracy])

history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 20, batch_size = 1024)

# As we can see the accuracy the model is start overfitting the model
# from epoch 7 as the accuracy remains constant around that accuracy

# model.fit(X_train, y_train, validation_split = 0.1, epochs = 5, batch_size = 512)

# Validation_data = In this we provide the seprate data for validation and
# Validation_split = In this the validation data is seprate out from the testing data


history_dict = history.history
print(history_dict)
print(history_dict.keys())

# =============================================================================
# score = model.evaluate(X_test, y_test)
# score
# =============================================================================


# Ploting the Training and Validation loss

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]

epochs = np.arange(0, len(loss_values))

def Loss_plot():
    plt.plot(epochs, loss_values, 'bo', label = 'Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label = 'Validation Loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

Loss_plot()

# Plotting the Training and Validation Accuracy

def Acc_plot():
    plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Loss')
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

Acc_plot()

''' As by seeing the graph after 7 epochs the model starts overfitting over
the training data '''

''' Hence retrain the model over the 7 epochs '''
''' Doing more tuning in this model '''


model1 = models.Sequential()
model1.add(layers.Dense(64, activation = 'tanh', input_shape = (1000, )))
model1.add(layers.Dense(128, activation = 'relu'))
model1.add(layers.Dense(32, activation = 'tanh'))
model1.add(layers.Dense(1, activation = 'sigmoid'))

model1.compile(optimizer = 'rmsprop', loss = losses.binary_crossentropy, metrics = ["accuracy"])

history1 = model1.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 6, batch_size = 512)

history_dict1 = history1.history
print(history_dict1)
print(history_dict1.keys())

results = model1.evaluate(X_test, y_test)
print(results)

y_pred = model1.predict(X_test)
print(y_pred)

''' Near to 0 - means negative
Near to 1 - means positive
Model is much confident about this prediction
but
for value like 0.4, 0.6 model is less confident '''

y_pred1 = np.around(y_pred)
print(y_pred1)


from findhypara import BestModel
bm = BestModel()
history = bm.findmodel(Partial_X_train, Partial_y_train, (1000, ), 1, "tanh", "relu", "sigmoid", "rmsprop", "binary_crossentropy", ["accuracy"], 6, "combination", X_val, y_val, [3], [64, 128], [64])
model_data = bm.savemodelhistory()
model_history = bm.savefullhistory()
values = bm.getmodel(1)
bm.loss_plot()
bm.acc_plot()
