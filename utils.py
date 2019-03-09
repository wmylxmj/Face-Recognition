# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:54:02 2019

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.misc
from glob import glob
import os
import random

class DataLoader(object):
    
    def __init__(self, dataset_path=r'.\datasets\lfw', image_height=256, image_width=256):
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width
        pass
    
    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    def find_anchor_folders(self, dataset_path):
        anchor_folders = []
        folders = list(os.listdir(dataset_path))
        for folder in folders:
            files = list(os.listdir(os.path.join(dataset_path, folder)))  
            n_files = 0
            for file in files:
                _, ext = os.path.splitext(file.lower())
                if ext == ".jpg" or ext == ".png":
                    n_files += 1
                    if n_files >= 2:        
                        anchor_folders.append(folder)
                        break
                    pass
                pass
            pass
        return anchor_folders
    
    def search_triplets(self, dataset_path, max_samples=2000):
        APN = []
        anchor_folders = self.find_anchor_folders(dataset_path)
        random.shuffle(anchor_folders)
        for anchor_folder in anchor_folders:
            AP = []
            anchor_files = list(os.listdir(os.path.join(dataset_path, anchor_folder)))
            random.shuffle(anchor_files)
            n_anchor_files = len(anchor_files)
            for i in range(n_anchor_files):
                _, ext = os.path.splitext(anchor_files[i].lower())
                if ext == ".jpg" or ext == ".png":
                    anchor_file = anchor_files[i]
                    # find positive
                    for j in range(i+1, n_anchor_files):
                        _, ext = os.path.splitext(anchor_files[j].lower())
                        if ext == ".jpg" or ext == ".png":
                            positive_file = anchor_files[j]
                            AP.append((os.path.join(dataset_path, anchor_folder, anchor_file), \
                                       os.path.join(dataset_path, anchor_folder, positive_file)))
                            pass
                        pass
                    pass
                pass
            if len(AP)>=10:
                AP = random.sample(AP, 10)
                pass
            # find negative
            N = []
            negative_folders = list(os.listdir(dataset_path))
            negative_folders.remove(anchor_folder)
            negative_folders = random.sample(negative_folders, len(negative_folders)//20)
            random.shuffle(negative_folders)
            n_anchor_files = len(AP)
            for negative_folder in negative_folders:
                negative_files = list(os.listdir(os.path.join(dataset_path, negative_folder)))
                random.shuffle(negative_files)
                for negative_file in negative_files:
                    _, ext = os.path.splitext(negative_file.lower())
                    if ext == ".jpg" or ext == ".png":
                        N.append(os.path.join(dataset_path, negative_folder, negative_file))
                        break
                    pass
                if len(N)>=n_anchor_files:
                    break
                pass
            random.shuffle(AP)
            random.shuffle(N)
            for i in range(len(AP)):
                anchor, positive = AP[i]
                negative = N[i]
                APN.append((anchor, positive, negative))
                if len(APN)>=max_samples:
                    return APN
                pass
            pass
        return APN
    
    def load_batches(self, batch_size=4, normalize=True, complete_batch_only=True):
        search_result = self.search_triplets(self.dataset_path)
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
                anchor_image = scipy.misc.imresize(anchor_image, (self.image_width, self.image_height))
                anchor_images.append(anchor_image)
                positive_image = self.imread(positive)
                positive_image = scipy.misc.imresize(positive_image, (self.image_width, self.image_height))
                positive_images.append(positive_image)
                negative_image = self.imread(negative)
                negative_image = scipy.misc.imresize(negative_image, (self.image_width, self.image_height))
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
                anchor_image = scipy.misc.imresize(anchor_image, (self.image_width, self.image_height))
                anchor_images.append(anchor_image)
                positive_image = self.imread(positive)
                positive_image = scipy.misc.imresize(positive_image, (self.image_width, self.image_height))
                positive_images.append(positive_image)
                negative_image = self.imread(negative)
                negative_image = scipy.misc.imresize(negative_image, (self.image_width, self.image_height))
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
