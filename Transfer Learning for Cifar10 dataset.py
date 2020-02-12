# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 20:19:03 2020

@author: Chirag
"""

# Using Transfer Learning in cifar-10 dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.datasets import cifar10

(train_data, train_label), (test_data, test_label) = cifar10.load_data()

train_data = train_data.astype("float32")


from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from keras.utils.np_utils import to_categorical

base_model = MobileNetV2(weights = "imagenet", include_top = False, input_shape = (32, 32, 3))

i = 0
for layers in base_model.layers:
    layers.trainable = False
    i = i + 1 
    if (i > 155): # Here all layers are non-trainable
        break

x=base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=preds)

y_sparse = to_categorical(train_label)

model.compile(optimizer = optimizers.RMSprop(learning_rate = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])

model.fit(train_data, y_sparse, epochs = 10, batch_size = 128)
