# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:31:39 2020

@author: Chirag
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing training set
train_data = pd.read_csv("Google_Stock_Price_Train.csv")
train_data = train_data.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
train_data = mm.fit_transform(train_data)

# Getting inputs and outputs
X_train = train_data[:-1]
y_train = train_data[1:]

# Reshaping (batch_size, timestamp, input_dim) timestamp = t - (t-1), input_dim = no. of features
X_train = np.reshape(X_train, (1257, 1, 1))

# Part - 2 Building RNN

# Importing Keras Libraries and Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Initializing the RNN
model = Sequential()

# Adding the input layer and LSTM layer
model.add(LSTM(4, activation = 'relu', input_shape = (None, 1)))

# Adding the Output Layer
model.add(Dense(1))

# Compile the RNN
model.compile('adam', loss = 'mse', metrics = ['mae'])

# Fitting the RNN to the training set
model.fit(X_train, y_train, batch_size = 32, epochs = 200)

# Part - 3 Making the PReditions and Visualising the results

# Getting the real stock price of 2017
test_data = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = test_data.iloc[:, 1:2].values

# Getting predicted of stock price 2017
inputs = real_stock_price
inputs = mm.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = model.predict(inputs)
predicted_stock_price = mm.inverse_transform(predicted_stock_price)

# Visualising the Results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


# Homework

# Getting the Real Stock Price of 2012 - 2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values

# Getting the Predicted Stock Price of 2012 - 2016
predicted_stock_price_train = model.predict(X_train)
predicted_stock_price_train = mm.inverse_transform(predicted_stock_price_train)

# Visualising the Results
plt.plot(real_stock_price_train, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price_train, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Sock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google stock Price')
plt.legend()
plt.show()

# Part - 4 Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
rmse/800

