# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:08:59 2020

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

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

import imutils

import cv2
from sklearn.metrics.cluster import entropy
from skimage import io, color, img_as_ubyte,measure
import skimage
import os
import glob

from sklearn.neighbors import KNeighborsClassifier
from dominantcolors import get_image_dominant_colors

#model file name
filename = 'KNC classifier.sav'

loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)
print(loaded_model.classes_)
#image file name
file= "testing/5/5m (8).jpg"



# read the training image
rgbImg = io.imread(file)
grayImg = img_as_ubyte(color.rgb2gray(rgbImg))
entrop1=entropy(grayImg)
myimg = cv2.imread(file)
output = imutils.resize(myimg, width=400)


dominant_colors = get_image_dominant_colors(image_path=file,num_colors=4)
entropy2 = measure.shannon_entropy(myimg)
final=np.array([entrop1,entropy2])
for x in np.nditer(dominant_colors):
	final=np.append(final,x)
prediction = loaded_model.predict(final.reshape(1,-1))
label = "{}".format(prediction)
cv2.putText(output, label, (10, (30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the probabilities for each of the individual labels
cv2.imshow("Output", output)
key = cv2.waitKey(0) & 0xFF

if key == ord("q"):
    cv2.destroyAllWindows()
		

        
print("Prediction - {}".format(prediction))
