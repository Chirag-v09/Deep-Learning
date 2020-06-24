# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:16:58 2020

@author: Chirag
"""

import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimentionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimentionality))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        print(j, word)
        index = abs(hash(word)) % dimentionality
        results[i ,j, index] = 1

