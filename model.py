# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:56:32 2019

@author: wmy
"""

import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, \
concatenate, AveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda, Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import pydot
from keras.initializers import glorot_uniform
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

def conv2d_block(X, filters, padding='same', strides=(1, 1)):
    f_3x3 = int(filters/8*3)
    f_5x5 = int(filters/8*3)
    f_7x7 = filters - f_3x3 - f_5x5
    X_3x3 = Conv2D(filters=f_3x3, kernel_size=(3, 3), strides=strides, \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X)
    X_5x5 = Conv2D(filters=f_5x5, kernel_size=(3, 3), strides=strides, \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X)
    X_5x5 = Conv2D(filters=f_5x5, kernel_size=(3, 3), strides=(1, 1), \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X_5x5)
    X_7x7 = Conv2D(filters=f_7x7, kernel_size=(3, 3), strides=strides, \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X)
    X_7x7 = Conv2D(filters=f_7x7, kernel_size=(3, 3), strides=(1, 1), \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X_7x7)
    X_7x7 = Conv2D(filters=f_7x7, kernel_size=(3, 3), strides=(1, 1), \
                   padding=padding, kernel_initializer=glorot_uniform(seed=0))(X_7x7)
    output = layers.concatenate([X_3x3, X_5x5, X_7x7], axis=3)
    return output

def convolutional_block(X, filters_list, strides=(2, 2)):
    f1, f2, f3 = filters_list
    X_shortcut = X
    X = Conv2D(f1, (1, 1), strides=strides, padding='valid', \
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = conv2d_block(X, filters=f2)
    X = BatchNormalization(axis=3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', \
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X_shortcut = Conv2D(f3, (1, 1), strides=strides, padding='valid', \
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    X = layers.add([X, X_shortcut])
    X = LeakyReLU(alpha=0.1)(X)
    return X

def res_block(X, filters_list):
    f1, f2, f3 = filters_list
    X_shortcut = X
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1,1), padding='valid', \
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = conv2d_block(X, filters=f2)
    X = BatchNormalization(axis=3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Conv2D(f3, (1, 1), strides=(1, 1), padding='valid', \
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = layers.add([X, X_shortcut])
    X = LeakyReLU(alpha=0.1)(X)
    return X

def encoding_body(X):
    X = Conv2D(64, (3, 3), strides=(1, 1), \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Conv2D(64, (3, 3), strides=(2, 2), \
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = convolutional_block(X, filters_list=[64, 64, 256])
    X = res_block(X, [64, 64, 256])
    X = res_block(X, [64, 64, 256])
    X = convolutional_block(X, filters_list=[128, 128, 512])
    X = res_block(X, [128, 128, 512])
    X = res_block(X, [128, 128, 512])
    X = res_block(X, [128, 128, 512])
    X = convolutional_block(X, filters_list=[256, 256, 1024])
    X = res_block(X, [256, 256, 1024])
    X = res_block(X, [256, 256, 1024])
    X = res_block(X, [256, 256, 1024])
    X = res_block(X, [256, 256, 1024])
    X = res_block(X, [256, 256, 1024])
    X = convolutional_block(X, filters_list=[512, 512, 2048])
    X = res_block(X, [512, 512, 2048])
    X = res_block(X, [512, 512, 2048])
    X = AveragePooling2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    X = Dense(128, kernel_initializer=glorot_uniform(seed=0))(X)
    return X

def SiameseNetwork(input_shape):
    input_A = Input(input_shape)
    input_B = Input(input_shape)
    merged = concatenate([input_A, input_B], axis=0)
    encodings = encoding_body(merged)
    batch_size = tf.floordiv(tf.shape(encodings)[0], 2)
    l1_layer = Lambda(lambda tensors:K.abs(tensors[0:batch_size] - tensors[batch_size:2*batch_size]))
    l1_distance = l1_layer(encodings)
    prediction = Dense(1, activation='sigmoid')(l1_distance)
    model = Model(inputs=[input_A, input_B], outputs=prediction)
    return model

