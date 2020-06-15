# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:27:24 2020

@author: Matador
"""

import pandas as pd 
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# # from sklearn.neighbors import KNeighborsClassifier
arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
              'APC','NEB','PB','UB','FPNB']

base = {}
data = []
label = []

for k in range(len(arritmias)):

    ds = pd.read_csv('feature_wavelet/'+arritmias[k]+'.csv')
    
    aux0 = len(ds.get('0').values)
    
    dat = np.zeros((aux0, 18))
    
    
    for j in range(18):
        
        aux = ds.get(str(j)).values
        
        for i in range(len(aux)):
            
            dat[i][j] = aux[i]
        
        for i in range(aux0):
            
            data.append(dat[i,:])
            label.append(k)


base['label'] = np.asarray(label)
data = np.asarray(data)

columns = []

for i in range(18):
    
    base['c'+str(i)] = data[:,i]
    columns.append('c'+str(i))
    

columns.append('label')
dv = pd.DataFrame(base , columns = columns)

dv.to_csv('my_features_and_labels.csv')       

    

# from sklearn.datasets import load_iris
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn import metrics
# # import matplotlib.pyplot as plt

# iris_dataset = load_iris()

# keys = iris_dataset.keys()

# data = iris_dataset['data']
# target = iris_dataset['target']
# target_name = iris_dataset['target_names']
        
# #     plt.figure()
# #     io.imshow(dat)      


