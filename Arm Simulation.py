# iGrasp Summer Training - Neural Network Model for GYM-HANDREACH pre-built environment.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import gym
import numpy as np

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


# define the keras model
model = Sequential()
model.add(Dense(32, input_dim=83, activation='relu')) # batch_size = 32 , input_dim = 20 (action) + 63 (observation)
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))               # added 3 layers between output and input
model.add(Dense(63, activation='sigmoid'))            # output is next observation

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='sgd')

env = gym.make("HandReach-v0")
observation = env.reset()

neuralnetworkinput = np.empty((1000000,83), dtype = np.float32)
nextobserves = np.empty((1000000,63), dtype = np.float32)

sampledonce = None


while True:
    
    for i in range(999999): # one million samples
        env.render()
  
        action = env.action_space.sample()  # your agent here (this takes random actions)
  
        neuralnetworkinput[i] = np.concatenate((observation["observation"],action))
  
        observation, reward, done, info = env.step(action)  # action has 20 elements
   
        nextobserves[i] = observation["observation"]  # observation has 63 elements
         
        if sampledonce:
            
            random_indices_train = np.random.randint(0, neuralnetworkinput.shape[0], size=32)
            
            x_batch = neuralnetworkinput[random_indices_train]
            y_batch = nextobserves[random_indices_train]
            
            random_indices_test = np.random.randint(0, nextobserves.shape[0], size=32)
            
            x_test = neuralnetworkinput[random_indices_test]
            y_test = nextobserves[random_indices_test]
            
            model.fit(x_batch, y_batch, callbacks=[tensorboard_callback])
            
        if i%1000 == 0:
           print("Test loss: ",model.test_on_batch(x_test,y_test))
           env.reset()
            
    sampledonce = True