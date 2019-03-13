# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:23:22 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
from glob import glob
import os
import random

class CASPEALR1DataLoader(object):
    
    def __init__(self, dataset_path=r'.\datasets\CAS-PEAL-R1', image_height=480, image_width=360):
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width
        pass
    
    def check_tiff(self, file):
        _, ext = os.path.splitext(file.lower())
        if ext == ".tif" or ext == ".tiff":
            return True
        else:
            return False
        pass
    
    def make_dir(self, path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
            print("You created a new path!")
            print("Path: " + str(path))
            pass
        else:
            print("Path: " + str(path) + " is already existed.")
        pass
    
    def write_infos(self, save_path=r'.\infos\CAS-PEAL-R1'):
        self.make_dir(save_path)
        self.infos_path = save_path
        numbers = os.listdir(self.dataset_path + r'\POSE')
        for number in numbers:
            file = open(save_path + '\\' + str(number) + '.txt', 'w')
            print('file: ' + save_path + '\\' + str(number) + '.txt' + ' created.')
            for tiff in os.listdir(self.dataset_path + r'\POSE' + '\\' + str(number)):
                if self.check_tiff(tiff):
                    file.write(self.dataset_path + r'\POSE' + '\\' + \
                               str(number) + '\\' + tiff + '\n')
                    pass
                pass
            file.close()
            pass
        for folder in os.listdir(self.dataset_path + r'\FRONTAL'):
            print('searching: ' + self.dataset_path + r'\FRONTAL' + '\\' + folder)
            for tiff in os.listdir(self.dataset_path + r'\FRONTAL' + '\\' + folder):
                if self.check_tiff(tiff):
                    number_str = tiff[3:9]
                    file = open(save_path + '\\' + number_str + '.txt', 'a+')
                    file.write(self.dataset_path + r'\FRONTAL' + '\\' + \
                               folder + '\\' + tiff + '\n')
                    file.close()
                    pass
                pass
            pass
        print('finished!')
        pass         
    
    def search_triplets(self, infos_path=r'.\infos\CAS-PEAL-R1', max_samples=2000, seed=0):
        APN = []
        random.seed(seed)
        files = list(os.listdir(infos_path))
        random.shuffle(files)
        for i in range(max_samples):
            anchor_index = random.randint(0, len(files)-1)   
            negative_index = random.randint(0, len(files)-1)
            while(anchor_index==negative_index):
                negative_index = random.randint(0, len(files)-1)
                pass
            anchor_txt = open(infos_path + '\\' + files[anchor_index], 'r')
            negative_txt = open(infos_path + '\\' + files[negative_index], 'r')
            tiffs_anchor = anchor_txt.readlines()
            tiffs_negative = negative_txt.readlines()
            anchor, positive = random.sample(tiffs_anchor, 2)
            negative = random.sample(tiffs_negative, 1)[0]
            anchor = anchor.strip()
            positive = positive.strip()
            negative = negative.strip()
            APN.append((anchor, positive, negative))
            pass
        return APN
    
    def imread(self, path):
        image = scipy.misc.imread(path, mode='L').astype(np.float)
        image = scipy.misc.imresize(image, (self.image_height, self.image_width))
        image = image[:, :, np.newaxis]
        return image
    
    def load_batches(self, batch_size=4, normalize=True, complete_batch_only=True, seed=0):
        random.seed(seed)
        search_result = self.search_triplets(seed=seed)
        random.shuffle(search_result)
        self.n_complete_batches = int(len(search_result) / batch_size)
        self.n_batches = int(len(search_result) / batch_size)
        if ((len(search_result)/batch_size) > int(len(search_result)/batch_size)) and \
        complete_batch_only==False:
            self.n_batches += 1
            pass
        for i in range(self.n_complete_batches):
            batch = search_result[i*batch_size:(i+1)*batch_size]
            anchor_images, positive_images, negative_images = [], [], []
            for anchor, positive, negative in batch:
                anchor_image = self.imread(anchor)
                anchor_images.append(anchor_image)
                positive_image = self.imread(positive)
                positive_images.append(positive_image)
                negative_image = self.imread(negative)     
                negative_images.append(negative_image)
                pass
            if normalize:
                anchor_images = np.array(anchor_images)/127.5 - 1.0
                positive_images = np.array(positive_images)/127.5 - 1.0
                negative_images = np.array(negative_images)/127.5 - 1.0
                pass
            yield anchor_images, positive_images, negative_images                        
        if self.n_batches > self.n_complete_batches:
            batch = search_result[self.n_complete_batches*batch_size:]
            anchor_images, positive_images, negative_images = [], [], []
            for anchor, positive, negative in batch:
                anchor_image = self.imread(anchor)
                anchor_images.append(anchor_image)
                positive_image = self.imread(positive)
                positive_images.append(positive_image)
                negative_image = self.imread(negative)
                negative_images.append(negative_image)
                pass
            if normalize:
                anchor_images = np.array(anchor_images)/127.5 - 1.0
                positive_images = np.array(positive_images)/127.5 - 1.0
                negative_images = np.array(negative_images)/127.5 - 1.0
                pass
            yield anchor_images, positive_images, negative_images        
        pass
            
    pass

