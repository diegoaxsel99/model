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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from sklearn.decomposition import PCA
"""
funciones para extraer la informacion de los archivos creados donde se
encuentran las caracteristicas
"""
def get(tipo,random,pca):
    
    if(pca):
        path = 'matriz/pca_features.csv'
    else:
        path = 'matriz/all_features.csv'
    
    dn = pd.read_csv(path)
    
    
    [f, c] = dn.shape
    c = c - 1
    
    target = dn.get('label').values
    columns = dn.columns
    data = dn.iloc[:,:c].values   
    
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
    
    if(tipo == 'Dataframe'):
         dnn = dn.iloc[:, :c]
         return dnn
        
    
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

def plot_corr(df , size = 20):
    
    corr = df.corr()
    corr_v = corr.iloc[:,:].values
    
    fig, ax = plt.subplots(figsize = (size,size))
    cmap = cm.viridis
    
    plt.matshow(corr_v , cmap = cmap)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.colorbar()

def visualizacion(data,target):
    
    pca = PCA(n_components = 2)
    pca.fit(data)
    dff = pd.DataFrame(pca.transform(data), columns = ['pc1', 'pc2'])
    
    arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
                  'APC','NEB','PB','UB','FPNB','ROTPVC']
    
    
    ntarget = []
    
    for i in range(len(target)):
        print(target[i])
        
        ntarget.append(arritmias[target[i]])
        
    dt = pd.DataFrame(ntarget, columns = ['target'])
    dfinal = pd.concat([dff,dt], axis = 1)
    
    colors = ['b','r','g','c','m','y','k','w','']
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('pc1', fontsize = 15)
    ax.set_ylabel('pc2', fontsize = 15)
    ax.set_title('2 component PCA para visualizar ', fontsize = 20)
    
    for target, color in zip(arritmias,colors):
        
       indicesToKeep = dfinal['target'] == target
       ax.scatter(dfinal.loc[indicesToKeep, 'pc1']
                  ,dfinal.loc[indicesToKeep, 'pc2']
                  , c = color
                  , s = 50) 
       
    ax.legend(arritmias)
    ax.grid()
        
    