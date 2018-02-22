#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:19:42 2018

@author: marta
"""

import glob
import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imresize
from keras.layers import Dense, Dropout,   Flatten,Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from PIL import Image


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

file_dir = 'database1/'


# Image settings
h = 100  # height
w = 100 # width
dp = 3  # color RGB
#classes = ['crime', 'culture', 'disasters','food','gossip' ,'health' ,'movie', 'nature' ,'politics','sports','tec']
classes = ['normal', 'outlier']


def readimage(directory):
    
    img_paths = glob.glob(directory + '**/*.jpg', recursive=True)
    n_imgs = len(img_paths) # number of images
    imgs = np.zeros((n_imgs,h,w,dp), dtype=np.float32)
    
    
    listfile =[]
    listot =[]
    labels=[]
    j=0
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            addrs = sorted(glob.glob(root + dirname + '/*.jpg'))
            for i, val in enumerate(addrs):
                #labels.append(len(listfile)+1)
                labels.append(dirname)
                img = imread(val)
                
                if len(img.shape)==2:
                    img = Image.open(val).convert('RGB')
                    
                
                img = imresize(img,(h,w))    
                imgs[j, ...] = img
                j=j+1
            listfile.append(addrs)
            print('read files:' + root + dirname)
    
    #labels=LabelBinarizer().fit_transform(labels)        
    for item in listfile:
        listot = listot + item
        
    return imgs,labels





def build_cnn():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(h, w, dp))) # 200x150x3 => 198x148x32
      
    model.add(MaxPooling2D(pool_size=(2, 2))) # 198x148x32 => 99x74x32
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 200x150 => 100x75
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5 , name='dropout2'))
    model.add(Dense(len(classes), activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print('Loading file training and test set')
X,y = readimage(file_dir)


from sklearn.model_selection import train_test_split
print('Spliting training and test set')
#CONFERIR AS LISTAS
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=7)

########################### ENCODING CLASSES ################################
### Change any labels to sequential integer labels
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
Yn_train = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
Yi_train = np_utils.to_categorical(Yn_train)
uniques, ids = np.unique(Yn_train, return_inverse=True)
n_classes = len(uniques)
Yn_test = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
Yi_test = np_utils.to_categorical(Yn_test)

print(n_classes)
print(uniques)
########################### NORMALIZATION ####################################
X_train /= 255 # ou gaussian
X_test /= 255
# Preprocess input data
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Preprocess class labels
Y_train = Yi_train
Y_test = Yi_test

model = build_cnn()

print('Training')

model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)


print('Testing')
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)


# calculate predictions
predictions = model.predict(X_test)
proba = model.predict_proba(X_test)
Y_predict = model.predict_classes(X_test)



print(score)

print('writing result file')

import json

with open('img_pred_percent.txt', 'w') as f:
    f.write(json.dumps(proba.tolist()))

with open('img_pred_class.txt', 'w') as f1:
    f1.write(json.dumps(Y_predict.tolist()))

with open('img_ground_truth.txt', 'w') as f2:
    f2.write(json.dumps(Yn_test.tolist()))
  