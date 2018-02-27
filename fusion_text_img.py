#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:27:30 2018

@author: marta
"""
from collections import Counter
import json
import numpy as np
#print acuracy /precision ReduceLROnPlateau
#https://stackoverflow.com/questions/45378493/why-does-a-binary-keras-cnn-always-predict-1
    
def load():
    
    #Now read the file back into a Python list object
    with open('img_ground_truth.txt', 'r') as f:
        img_ground_truth = json.loads(f.read())
        
    with open('img_pred_class.txt', 'r') as f:
        img_pred_class = json.loads(f.read())
        
    
    with open('img_pred_percent.txt', 'r') as f:
        img_pred_percent = json.loads(f.read())
        
    
    with open('txt_ground_truth.txt', 'r') as f:
        txt_ground_truth = json.loads(f.read())
        
    
    with open('txt_pred_class.txt', 'r') as f:
        txt_pred_class = json.loads(f.read())
        
    
    with open('txt_pred_percent.txt', 'r') as f:
        txt_pred_percent = json.loads(f.read())
    
    return img_ground_truth,img_pred_class,img_pred_percent,txt_ground_truth,txt_pred_class,txt_pred_percent

def swap(old_list):
    #swapp column    
    newlist=[]
    for i,x in enumerate(old_list):
        newlist.append([x[1],x[0]])

    return  newlist

def Metrics(lst_ground_truth,lst_pred):
    tpr=0
    fpr=0
    acc=0
    lst_cont_ground= Counter(lst_ground_truth)
    lst_cont_pred= Counter(lst_pred)
    
    # declared outlier intercecsion with ground truth outlier (true positive)
    for i,x in enumerate(lst_ground_truth):
        if (lst_pred[i]==0 and x==0):
            tpr=tpr+1
            
    for i,x in enumerate(lst_ground_truth):
        if (lst_pred[i]==0 and x==1):
            fpr=fpr+1
    
    #Accuracy
    for i,x in enumerate(lst_ground_truth):
        if (lst_pred[i]==x):
            acc=acc+1
    
    
    
    Precision=0
    Recall=0
    Accuracy=0
    
    if lst_cont_pred[0] !=0:             
        Precision= 100 * ( tpr/lst_cont_pred[0])
        
    if lst_cont_ground[0] !=0:    
        Recall= 100 * ( tpr/lst_cont_ground[0])
        
    if lst_cont_ground[0] !=0:    
        Accuracy= 100 * ( acc/(lst_cont_ground[0] + lst_cont_ground[1]))
    
    fpr_final= fpr/lst_cont_ground[0]
    
    return Precision,Recall,fpr_final,Accuracy

def print_err():
    
    for i,x in enumerate(txt_ground_truth):
        if(x!=txt_pred_class[i]):
            print('erro txt - num: ' + str(i))
        
        if(x!=img_pred_class[i]):
            print('erro img - num: ' + str(i))
    

def fusion(img_pred, txt_pred):
    final_class=[]
    """
    if Recall_img > Recall_txt:
        peso_img =2
        peso_txt=1
    else:
        peso_img=1
        peso_txt=2
     """   
    for i,x in enumerate(img_pred):
        #item= np.sum([[img_percent[i][0] *peso_img ,img_percent[i][1] * peso_img],[txt_percent[i][0] *peso_txt ,txt_percent[i][1] * peso_txt]], axis=0)/2
        #maior=np.amax(item)
        #if maior== item[0]:
        #    final_class.append(1)
        #else:            
        #    final_class.append(0)
        if x==0 or txt_pred[i]==0:
            final_class.append(0)
        else:
            final_class.append(1)
            
    return final_class
        
        
            
    
img_ground_truth,img_pred_class,img_pred_percent,txt_ground_truth,txt_pred_class,txt_pred_percent =load()
#txt_pred_percent =swap(np.asarray(txt_pred_percent))

#print_err()

Precision_img,Recall_img,fpr_final_img,Accuracy_img=Metrics(img_ground_truth,img_pred_class)

Precision_txt,Recall_txt,fpr_final_txt,Accuracy_txt=Metrics(txt_ground_truth,txt_pred_class)



#FUSION
final_class= fusion(img_pred_class,txt_pred_class)

Precision,Recall,fpr_final,Accuracy_final=Metrics(txt_ground_truth,final_class)



    
    