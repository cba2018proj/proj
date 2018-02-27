#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:19:42 2018

@author: marta
"""

import glob
import numpy as np
import os
from scipy.misc import imresize
from keras.layers import Dense, Dropout,   Flatten,Conv2D, MaxPooling2D,BatchNormalization,Activation
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from PIL import Image
from keras import optimizers
import imageio


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

file_dir = 'database1a/'


# Image settings
h = 224  # height
w = 224 # width
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
    filename=[]
    j=0
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            addrs = sorted(glob.glob(root + dirname + '/*.jpg'))
            
            for i, val in enumerate(addrs):
                #labels.append(len(listfile)+1)
                filename.append(os.path.basename(val))
                labels.append(dirname)
              
                img = imageio.imread(val)
                
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
        
    return imgs,labels,filename



def build_cnn_alex():
    """"
    The first convolutional layer filters the 224×224×3 input image with 96 kernels of size 11×11×3
    with a stride of 4 pixels (this is the distance between the receptive field centers of neighboring
    neurons in a kernel map). The second convolutional layer takes as input the (response-normalized
    and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48.
    The third, fourth, and fifth convolutional layers are connected to one another without any intervening
    pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 ×
    256 connected to the (normalized, pooled) outputs of the second convolutional layer. The fourth
    convolutional layer has 384 kernels of size 3 × 3 × 192 , and the fifth convolutional layer has 256
    kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each.

    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    """

    # Define model architecture
    model = Sequential()

    #AlexNet with batch normalization in Keras 
    #input image is 224x224

    model.add(Conv2D(64, (11, 11), input_shape=(h, w, dp)))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(128, (7, 7)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(192, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(len(classes), kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # Compile model
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


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
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


print('Loading file training and test set')
X,y,filenames = readimage(file_dir)


from sklearn.model_selection import train_test_split
print('Spliting training and test set')
#CONFERIR AS LISTAS
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=7)
filenames_train, filenames_test,y_filename_train, y_filename_test = train_test_split( filenames, y, test_size=0.20, random_state=7)

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


model = build_cnn_alex()

print('Training')


model.fit(X_train, Y_train,  batch_size=32, epochs=20, verbose=1)

print('Testing')
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)


# calculate predictions
predictions = model.predict(X_test)
proba = model.predict_proba(X_test)
Y_predict = model.predict_classes(X_test)

Y_predict = [1 if (x==0) else 0 for x in Y_predict]
Yn_test = [1 if (x==0) else 0 for x in Yn_test]

   
print(score)

print('writing result file')



import json
 
with open('img_pred_percent.txt', 'w') as f:
    f.write(json.dumps(proba.tolist()))

with open('img_pred_class.txt', 'w') as f1:
    f1.write(json.dumps(Y_predict))

with open('img_ground_truth.txt', 'w') as f2:
    f2.write(json.dumps(Yn_test))

"""
#GRAFICO------------------

from keras import backend as K



get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[8].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X_test, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X_test, 1])[0]


from sklearn.decomposition import PCA

#http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html
#key: view PCA python
pca = PCA(n_components=3)
pca.fit(layer_output)
layer_output3 = pca.transform(layer_output)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



centers = [[1, 1], [-1, -1], [1, -1]]

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)


for name, label in [('normal', 0), ('outlier', 1)]:
    ax.text3D(layer_output3[Yn_test == label, 0].mean(),
              layer_output3[Yn_test == label,  1].mean() + 1.5,
              layer_output3[Yn_test == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
Yn_test = np.choose(Yn_test, [2, 1]).astype(np.float)



ax.scatter(layer_output3[:, 0], layer_output3[:, 1], layer_output3[:, 2], c=Yn_test, 
           edgecolor='k')


ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.legend()
plt.show()



pca = PCA(n_components=2)
pca.fit(layer_output)
layer_output2 = pca.transform(layer_output)


# Percentage of variance explained for each components
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

plt.figure()
for c, i, target_name in zip("rgb", [2, 1],  ['normal','outlier']):
   plt.scatter(layer_output2[Yn_test==i,0], layer_output2[Yn_test==i,1], c=c, label= target_name)
   
   

for i, txt in enumerate(layer_output2):
    plt.annotate(i, (layer_output2[i,0],layer_output2[i,1]))
    
    
    
plt.legend()
plt.title('PCA 2D')

plt.show()

"""
  