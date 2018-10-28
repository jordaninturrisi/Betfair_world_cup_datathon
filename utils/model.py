import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, Flatten, Reshape
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization, GaussianNoise
from keras.regularizers import l2
from keras.initializers import TruncatedNormal

# Build model
def linear_model(num_inputs):
    model = Sequential()
    model.add(Dense(1, activation='linear', input_dim=num_inputs))
    model.add(Dense(3, activation='softmax'))
    return model

def tanh_model(num_inputs):
    model = Sequential()
    model.add(Dense(1, activation='tanh', input_dim=num_inputs))
    model.add(Dense(3, activation='softmax'))
    return model

def sigmoid_model(num_inputs):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=num_inputs))
    model.add(Dense(3, activation='softmax'))
    return model

def softmax_model(num_inputs):
    model = Sequential()
    model.add(Dense(3, activation='softmax', input_dim=num_inputs))
    return model

def relu_model(num_inputs):
    model = Sequential()
    model.add(Dense(100, input_dim=num_inputs))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax'))
    return model

def CNN_model(num_inputs):
    model = Sequential()

    model.add(Dense(100, input_dim=num_inputs))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.1))

    model.add(Reshape(target_shape=(5,5,4)))

    model.add(Conv2D(8, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.1))

    model.add(Conv2D(16, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(GaussianNoise(0.1))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(GaussianNoise(0.1))

    model.add(Dense(3, activation='softmax', kernel_regularizer=l2(1e-3)))

    return model
