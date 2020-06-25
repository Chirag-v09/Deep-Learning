# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:55:22 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image

from tensorflow.keras import models

model = tf.keras.models.load_model("sld_model.h5")
model.summary()

layers_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs = model.input, outputs = layers_outputs)
activation_model.summary()

i = 0
for layer in model.layers:
    i = i+1
    print(layer)

img = image.load_img("C_test.jpg", target_size = (224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor = img_tensor / 255.
img_tensor.shape

plt.imshow(img_tensor[0])
plt.show()

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 2], cmap = 'viridis')
plt.show()