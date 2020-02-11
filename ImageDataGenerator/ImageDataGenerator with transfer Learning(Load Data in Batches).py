
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model


train_data_gen = ImageDataGenerator(
                                rescale = 1.0/255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

test_data_gen = ImageDataGenerator(
                                rescale = 1.0/255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)


train_generator = train_data_gen.flow_from_directory("SubsetVMMR",
                                                     target_size = (224, 224),
                                                     # colour_mode = 'rgb'
                                                     batch_size = 32,
                                                     class_mode = 'categorical')

test_generator = test_data_gen.flow_from_directory("SubsetVMMR 1",
                                                   target_size = (224, 224),
                                                   # colour_mode = 'rgb'
                                                   batch_size = 32,
                                                   class_mode = 'categorical')





base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3)) # To remove the top bias layer


for layer in model.layers:
	layer.trainable = False

#   -------------
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(53,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)


# add new classifier layers
flat1 = Flatten()(base_model.outputs)
class1 = Dense(16, activation = 'relu')(flat1)
x = Dropout(0.4)(class1)
x = Dense(2, activation = 'relu')(x)
output = Dense(53, activation='softmax')(x)
# define new model
base_model = Model(inputs = base_model.inputs, outputs = output)


x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# =============================================================================
# model.add(Dense(1024, activation='tanh'))
# model.add(Dropout(0.5))
# =============================================================================


model.compile('adam', loss = "categorical_crossentropy", metrics = ["accuracy"])

model.fit_generator(train_generator, steps_per_epoch = 2, validation_data = test_generator, validation_steps = 2)

base_model.save('video_model.h5')
