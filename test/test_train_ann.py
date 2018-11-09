#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:56:43 2018
@author0: MIUKE
@author1: FS
"""
import sys
sys.path.append('../../entangl')
sys.path.append('../../dataset/')
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from entangl.data import data_bipartite as data


## Getting the data
(x_train, y_train, st_train), (x_test, y_test, st_test) = data.load_data_set('perfect_10k2k', states = True)
x_train = x_train[:, :3]
x_test = x_test[:, :3]
y_train /= np.log(2) 
y_test /= np.log(2) 
plt.hist(y_train)
plt.hist(y_test)


##Defining the ANN
batch_size = 100
nb_epoch = 2000
model = Sequential()
input_neurons = 3
hl_neurons = 30
hl2_neurons = 30
hl3_neurons = 30
initializer='normal'
#initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)  
model.add(Dense(units=hl_neurons, input_dim=input_neurons, kernel_initializer=initializer))
model.add(Activation('elu'))
#model.add(Dropout(0.2))
model.add(Dense(units=hl2_neurons, input_dim=hl_neurons, kernel_initializer=initializer))
model.add(Activation('elu'))
#model.add(Dropout(0.2))
model.add(Dense(units=hl3_neurons, input_dim=hl2_neurons, kernel_initializer=initializer))
model.add(Activation('elu'))
model.add(Dense(units=1, input_dim=hl3_neurons, kernel_initializer=initializer))
model.add(Activation('linear'))
model.compile(optimizer='adadelta',loss='mse', metrics=['mae'])
model.summary()


## Training it
history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1)

## Evaluating the ANN
evaluation_train = model.evaluate(x_train, y_train)
print("On training data [mse, mae]:")
print(evaluation_train)

evaluation = model.evaluate(x_test, y_test)
print("On test data [mse, mae]:")
print(evaluation)


## Visualization 
# 1: mae errors; 
# 2: hist of mae
# 3 & 4: weights of layer 1 visualized 
y_pred = np.squeeze(model.predict(x_test))   
err = np.abs(y_pred - y_test)
err_avg = np.average(err)
err_std = np.std(err)
y_pred_train = np.squeeze(model.predict(x_train))
err_train = np.abs(y_pred_train - y_train)
err_avg_train = np.average(err_train)
err_std_train = np.std(err_train)

plt.figure(1)
label1='avg(std) abs error - test: {:.3f}({:.3f}), \n train {:.3f} ({:.3f})'.format(
         err_avg, err_std, err_avg_train, err_std_train)
plt.plot(y_test, err, '.', label = label1)
plt.xlabel('Real entanglement values')
plt.ylabel('Abs error of')
plt.legend()

plt.figure(2)
plt.hist(err, bins=100)
plt.show(2)

