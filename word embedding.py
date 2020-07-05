# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:44:29 2020

@author: Chirag
"""

'''
1) Sentenses
2) one hot representation -> index from the dictionary
3) onhot repre -> embdding layer keras to form embedding matrix
4) embedding matrix

vocab size = 10,000; dimension = 10
'''

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sent = ['the glass of milk',
        'the glass of juice',
        'the cup of tea',
        'I am a good boy',
        'I am a good developer',
        'understand the meaning of word',
        'your videos are good']

# this is my vocabulary size:
voc_size = 10000

onehot_repr = [one_hot(word, voc_size) for word in sent]


from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sent_length = 8
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_length)

dim = 10

model = Sequential()
model.add(Embedding(voc_size, 10, input_length = sent_length))
model.compile('adam', loss = 'mse', metrics = ["accuracy"])

model.summary()

print(model.predict(embedded_docs))

embedded_docs[0]

print(model.predict(embedded_docs)[0])
