'''   MNIST DATASETS   '''

# Getting the data
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping (Data Preprocessing)
train_images = train_images.reshape(60000,784)
test_images = test_images.reshape(10000,784)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

from keras import models
from keras import layers

# Getting the model
model = models.Sequential()
# Adding Some Layers
model.add(layers.Dense(32, activation = 'relu', input_shape = (784, )))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 10, batch_size = 64, validation_split = 0.1)

# To save the Keras Model of name 'file.h5'
model.save_weights('file.h5')

# Load the Keras Model of name 'file.h5'
model.load_weights('file.h5')
