# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:35:52 2020

@author: Chirag
"""

# Single Image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# Taking Mahindra car image here
from tensorflow.keras.preprocessing.image import ImageDataGenerator\
, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
                                rescale = 1./255,
                                rotation_range = 40,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

img = load_img("A/A1.jpg")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 10, save_to_dir = 'ab generate', save_format = 'jpg'):
    i += 1
    if i>20:
        break


