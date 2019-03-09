# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:54:17 2019

@author: wmy
"""

import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from PIL import Image
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from utils import DataLoader
from model import SiameseNetwork

class FaceRecognition(object):
    
    def __init__(self, input_shape=(224, 224, 3), name=None):
        self.input_shape = input_shape
        self.image_height, self.image_width, _ = input_shape
        self.data_loader = DataLoader(image_height=self.image_height, \
                                      image_width=self.image_width)
        self.siamese_network = SiameseNetwork(input_shape)
        self.siamese_network.compile(loss='binary_crossentropy', \
                                     optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.name = name
        pass
    
    def train(self, epochs=1000, batch_size=3, load_pretrained=False):
        if load_pretrained:
            self.siamese_network.load_weights('./weights/siamese_network_weights.h5')
            print('Info: weights loaded.')
            pass
        for epoch in range(epochs):
            for batch_i, (anchor_images, positive_images, negative_images) \
            in enumerate(self.data_loader.load_batches(batch_size)):
                images_A = np.concatenate((anchor_images, anchor_images), axis=0)
                images_B = np.concatenate((positive_images, negative_images), axis=0)
                y_true = np.concatenate((np.ones((batch_size, 1)), np.zeros((batch_size, 1))), axis=0) 
                loss, accuracy = self.siamese_network.train_on_batch([images_A, images_B], y_true)
                print('[epoch: {0:}/{1:}][batch: {2:}/{3:}][loss: {4:}][accuracy: {5:}]'.format(epoch+1, \
                      epochs, batch_i+1, self.data_loader.n_batches, loss, accuracy))
                if (batch_i+1)%250==0:
                    self.siamese_network.save_weights('./weights/siamese_network_weights.h5')
                    print('Info: weights saved.')
                    pass
                pass
            if (epoch+1)%10==0:
                self.siamese_network.save_weights('./weights/siamese_network_weights.h5')
                print('Info: weights saved.')
                pass
            pass
        pass
    
    pass

fr = FaceRecognition()
fr.train(load_pretrained=True)