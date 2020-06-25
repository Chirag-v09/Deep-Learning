# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:54:55 2020

@author: Chirag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('DL Krish/fake-news/train.csv')
df.head()
df.iloc[0]

# Remove the rows having nan values
df.isnull().sum()
df = df.dropna()

X = df.drop('label', axis = 1)
# y = df.iloc[:, -1].values.reshape(-1, 1)
y = df['label']

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential
from tqdm import tqdm

voc_size = 10000

messages = X.copy()
messages['title'][1]
messages.reset_index(inplace = True)


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords

# Data Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []

for i in tqdm(range(len(messages))):
    review = re.sub('^a-zA-Z', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[:5]

onehot_repr = [one_hot(words, voc_size) for words in corpus]
onehot_repr[:5]

# Embedding Representation
sent_len = 20
embedded_docs = pad_sequences(onehot_repr, padding = 'pre', maxlen = sent_len)
embedded_docs[:5]

# Creating Model
embedded_vector_features = 50
model = Sequential()
model.add(Embedding(voc_size, embedded_vector_features, input_length = sent_len))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'sigmoid'))
model.compile('adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

X = np.asarray(embedded_docs)
y = np.asarray(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)


# Model training
model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 128, epochs = 10)

# Performance Metrics and Accuracy
y_pred = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

