# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:23:58 2020

@author: Matador
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler
"""
funciones para extraer la informacion de los archivos creados donde se
encuentran las caracteristicas
"""
def get(tipo,random):
    
    # ds = pd.read_csv('matriz/stats_features.csv')
    # dw = pd.read_csv('matriz/coeff_features.csv')
    # dx = pd.read_csv('matriz/time_features.csv')
    # ds.drop(['label'], axis='columns', inplace=True)
    
    # df = pd.concat([ds,dw], axis = 1)
    # df = pd.concat([df,dx], axis = 1)
    
    # [f, co] = df.shape 
    
    # co = co - 1
    # target = df.get('label').values
    
    # c = []
    # for i in range(co): c.append('c'+str(i))
    
    # d = []
    
    # for i in range(co): 
    #     aux = df[c[i]].values
    #     d.append(aux)
    
    # data = np.zeros((f,co))
    
    # for i in range(co):
        
    #     data[:,i] = d[i]
    
    # ds = pd.read_csv('matriz/stats_features.csv')
    # dw = pd.read_csv('matriz/coeff_features.csv')
    # dx = pd.read_csv('matriz/time_features.csv')
    # ds.drop(['label'], axis='columns', inplace=True)
    
    # dn = pd.concat([ds,dx], axis = 1)
    # dn = pd.concat([dn,dw], axis = 1)
    
    dn = pd.read_csv('matriz/all_features.csv')
    [f, c] = dn.shape
    c = c - 1
    
    data  = np.zeros((f,c))
    target = dn.get('label').values
    columns = dn.columns

    for i in range(len(columns) - 1):
        
        data[:,i] = dn.get(columns[i]).values
     
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=random ,test_size = 0.2)
    
    if(tipo == "cross"):
        
        return X_train, X_test, y_train, y_test
    
    if(tipo == "data"):
        return data, target
    
    if(tipo == "standardscaler"):
        
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        
        return X_train, X_test, y_train, y_test
        
    
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
    