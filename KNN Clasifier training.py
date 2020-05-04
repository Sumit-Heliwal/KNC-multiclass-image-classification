# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:56:49 2020

@author: Sumit
"""
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import mahotas as mt

import time

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

import cv2
from sklearn.metrics.cluster import entropy
from skimage import io, color, img_as_ubyte,measure
import skimage
import os
import glob

from sklearn.neighbors import KNeighborsClassifier
from dominantcolors import get_image_dominant_colors



neigh = KNeighborsClassifier()
train_path  = "train data"
train_names = os.listdir(train_path)
train_features = []
train_labels   = []
t1=time.time()

# loop over the training dataset
print("[STATUS] Started extracting ..")
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1
	for file in glob.glob(cur_path + "/*.jpg"):
		print("Processing Image - {} in {}".format(i, cur_label))
		# read the training image
		rgbImg = io.imread(file)
		grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
		entrop1=entropy(grayImg)
		myimg = cv2.imread(file)
		dominant_colors = get_image_dominant_colors(image_path=file,num_colors=4)
		entropy2 = measure.shannon_entropy(myimg)
		final=np.array([entrop1,entropy2])
		for x in np.nditer(dominant_colors):
			final=np.append(final,x)
		train_features.append(final)                        
		train_labels.append(cur_label)
		
		i += 1
        
print("[STATUS] Creating the classifier..")
print("[STATUS] Fitting data/label to model..")
neigh.fit(train_features, train_labels)
t2=time.time()
print('time for training is ',t2-t1 )
filename = 'KNC classifier.sav'
pickle.dump(neigh, open(filename, 'wb'))
