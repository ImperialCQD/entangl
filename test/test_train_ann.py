#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 23:56:43 2018

@author: FS
"""
import sys
sys.path.append('../../entangl')
sys.path.append('../../dataset')
import numpy as np
import matplotlib.pylab as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# import keras.optimizers as ko
# from keras.datasets import mnist
# from keras.utils import np_utils
# import keras.initializers
# from keras.callbacks import EarlyStopping, ModelCheckpoint

from entangl.data import data_bipartite as data

(x_train, y_train, st_train), (x_test, y_test, st_test) = data.load_data_set('perfect_10k2k', states = True)




## Getting the data
(x_train, y_train), (x_test, y_test) = data.load_data_set('test') 
y_train /= np.log(2) # normalization
y_test /= np.log(2) # normalization

##Defining the ANN
batch_size = 100
nb_epoch = 300

model = Sequential()
input_neurons=6
hl_neurons = 15
hl2_neurons = 15
hl3_neurons = 15
initializer='normal'
#initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)  

model.add(Dense(output_dim=hl_neurons, input_dim=input_neurons, init=initializer))
model.add(Activation('elu'))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=hl2_neurons, input_dim=hl_neurons, init=initializer))
model.add(Activation('elu'))
#model.add(Dropout(0.2))
model.add(Dense(output_dim=hl3_neurons, input_dim=hl2_neurons, init=initializer))
model.add(Activation('elu'))
model.add(Dense(output_dim=1, input_dim=hl3_neurons, init=initializer))
model.add(Activation('linear'))
model.compile(optimizer='adadelta',loss='mse', metrics=['mae'])
model.summary()
