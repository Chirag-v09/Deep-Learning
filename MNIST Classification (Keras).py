# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:04:36 2019

@author: Chirag
"""


'''   MNIST DATASETS   '''

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape = (784, )))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 10, batch_size = 64, validation_split = 0.1)

model.save_weights('file.h5')
model.load_weights('file.h5')

'''   55% - 16777216 - 3.41s   '''
'''   93% - 1024 - 4.98s   '''
'''   94% - 512 - 6s   '''
'''   95% - 256 - 6.87s   '''
'''   95% - 255 - 7s   '''
'''   95% - 128 - 9.65s   '''
'''   96% - 64 - 16s   '''
