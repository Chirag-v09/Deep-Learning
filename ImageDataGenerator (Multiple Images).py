
# Multile Image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import glob
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator\
, array_to_img, img_to_array, load_img

images = [cv2.imread(image) for image in glob.glob("chevrolet_impala_2007\*")]

images = images[:3]

for i in range(len(images)):
    images[i] = cv2.resize(images[i], (100, 100), interpolation = cv2.INTER_AREA)
    images[i] = images[i].reshape((1, ) + images[i].shape)



datagen = ImageDataGenerator(
                                rescale = 1./255,
                                rotation_range = 40,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)


for j in range(len(images)):
    i = 0
    for batch in datagen.flow(images, batch_size = 100, save_to_dir = 'a generate', save_format = 'jpg'):
        i += 1
        if i>2:
            break
