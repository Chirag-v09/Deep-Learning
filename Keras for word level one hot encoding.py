# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:18:35 2020

@author: Chirag
"""

from tensorflow.keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

#ss = ['cc aa bb', 'aa aa bb']

tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)
one_hot_results = tokenizer.texts_to_matrix(samples, mode = 'binary')

word_index = tokenizer.word_index
print('Fount %s unique tokens' %len(word_index))
