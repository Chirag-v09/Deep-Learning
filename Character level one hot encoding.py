# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:18:30 2020

@author: Chirag
"""

import numpy as np
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(range(1, len(characters) + 1), characters))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.keys()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        print(j, character)
        index = token_index.get(character)
        '''for key, value in token_index.items():
            if(value == character):
                index = key'''
        print(index)
        results[i, j, index] = 1


