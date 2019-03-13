# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:17:54 2019

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
from utils import CASPEALR1DataLoader
from model import SiameseNetwork
from facerec import FaceRecognition

fr = FaceRecognition()
fr.prepare()
fr.train(load_pretrained=True, seed=320)
