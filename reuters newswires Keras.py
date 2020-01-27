''' Single-label Multiclass Classification '''
''' 46 Different Topics i.e Classes '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing data :-
from keras.datasets import reuters
(train_data, train_label), (test_data, test_label) = reuters.load_data(num_words = 1000)

len(train_data)

def Vectorize_data(sequences, dimension = 1000):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        result[i, sequence] = 1
    return result

def Vectorize_label(labels, dimension = 46):
    result = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result


X_train = Vectorize_data(train_data).astype('float32')
X_test = Vectorize_data(test_data).astype('float32')

y_train = Vectorize_label(train_label).astype('float32')
y_test = Vectorize_label(test_label).astype('float32')

X_val = X_train[:1000]
partial_X_train = X_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]


from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import losses
from itertools import combinations

losses_list = []
val_loss_list = []
acc_list = []
val_acc_list = []

model = models.Sequential()
model.add(layers.Dense(512, activation = 'tanh', input_shape = (1000, )))
model.add(layers.Dense(1024, activation = 'relu'))
model.add(layers.Dense(46, activation = "softmax"))

model.compile('adam', loss = losses.categorical_crossentropy, metrics = ["accuracy"])

history = model.fit(partial_X_train, partial_y_train, batch_size = 512, epochs = 4, validation_data = (X_val, y_val))

history_dict = history.history
# print(history_dict.keys())

loss = history_dict['loss']
val_loss = history_dict['val_loss']
losses_list.append(loss)
val_loss_list.append(val_loss)

acc = history_dict['acc']
val_acc = history_dict['val_acc']
acc_list.append(acc)
val_acc_list.append(val_acc)

epochs = np.arange(0, len(acc))


# Ploting training and validation loss

def Loss_plot():
    plt.plot(epochs, loss, 'bo', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'b', label = 'validation loss')
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

Loss_plot()


# Ploting training and validation accuracy

def Acc_plot():
    plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

Acc_plot()


''' Predictions '''

predictions = model.predict(X_test)

np.sum(predictions[10])

''' It returns that index which has highest value i.e highest probability '''
np.argmax(predictions[0])



''' By using the sparse_catagorical_crossentropy
 i.e when we don't convert the y label into the sparse matrix '''


val_train_label = np.array(train_label[:1000])
partial_train_label = np.array(train_label[1000:])


model1 = models.Sequential()
model1.add(layers.Dense(512, activation = 'tanh', input_shape = (1000, )))
model1.add(layers.Dense(1024, activation = 'relu'))
model1.add(layers.Dense(46, activation = "softmax"))

model1.compile('adam', loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

history1 = model1.fit(partial_X_train, partial_train_label, batch_size = 512, epochs = 4, validation_data = (X_val, val_train_label))

predictions1 = model1.predict(X_test)




