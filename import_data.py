# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:23:58 2020

@author: Matador
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def get():
    
    df = pd.read_csv("my_features_and_labels.csv")
    
    target = df.get('label').values
    
    c = []
    for i in range(18): c.append('c'+str(i))
    
    d = []
    
    for i in range(18): 
        aux = df.get(c[i]).values
        d.append(aux)
    
    data = np.zeros((15606,18))
    
    for i in range(18):
        
        data[:,i] = d[i]
    
    return data,target

def coeff(X_test , y_test, model):
    
    y_pred = model.predict(X_test)

    matrix_c = confusion_matrix(y_test, y_pred)
    diag = np.diag(matrix_c)
    f = matrix_c[:,0]
    c = matrix_c[0,:] 
    
    
    tp = sum(diag) - diag[0]
    tn = diag[0]
    fp = sum(c) - c[0]
    fn = sum(f) - f[0]
    
    sens = tp /(tp + fn)
    esp =  tn /(tn + fp)
    
    return sens,esp
    


