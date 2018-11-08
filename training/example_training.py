import sys
sys.path.insert(0, '../entangl')
sys.path.insert(0, '../dataset')
from data import data_bipartite as data

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# import keras.optimizers as ko
# from keras.datasets import mnist
# from keras.utils import np_utils
# import keras.initializers
# from keras.callbacks import EarlyStopping, ModelCheckpoint


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


## Training the ANN
# early stopping can be enabled by uncommenting out es and 
# modifying fit see keras.callbacks for more details
# es = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')
history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=1)


## Evaluating the ANN
evaluation_train = model.evaluate(x_train, y_train, verbose=1)
print("On training data:")
print(evaluation_train)

evaluation = model.evaluate(x_test, y_test, verbose=1)
print("On test data:")
print(evaluation)


nn_tests = model.predict(x_test, batch_size=len(x_test))   # predicted entanglement values
nn_tests = nn_tests.reshape((len(nn_tests),))
actual_ys = y_test                                         # actual
msd = np.abs((nn_tests - actual_ys))
spread=np.average(msd)
spread2 = msd * msd  #spread squared
std = np.sqrt(np.abs(np.average(spread2) - spread*spread))




## Visualization 1: mae errors; 2: hist of mae; 3 & 4: weights of layer 1 visualized 
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(actual_ys, 
         msd, '.', label='mean absolute error test {0} std {1} Mean absolute error train {2} \n'.format(
        round(evaluation[1],3),round(std, 3), round(evaluation_train[1],3) ))
plt.xlabel('VN entanglement values')
plt.ylabel('Mean absolute error of NN vnes')
plt.legend()
plt.figure(2)
plt.hist(msd, bins=100)
plt.show(2)

x = model.get_weights()
plt.figure(3)
counter=0      
for i in range(hl_neurons):
    plt.plot(range(1,input_neurons+1), x[0][:,counter],'x-',label='HL neuron {0}'.format(counter+1))
    counter+=1
plt.xlabel('weights for 10 neurons in first hidden layer')
plt.ylabel('weights for 10 neurons in first hidden layer')
plt.xlabel('Input neuron number')
plt.legend()

plt.figure(4)
counter=0      
for i in range(input_neurons):
    plt.plot(range(1,hl_neurons+1), x[0][counter],'x-',label='Input neuron {0}'.format(counter+1))
    counter+=1
plt.ylabel('weights for 10 neurons in first hidden layer')
plt.xlabel('HL neuron number')
plt.legend()