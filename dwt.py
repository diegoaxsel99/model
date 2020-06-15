# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:30:41 2020

@author: Matador
"""

import pywt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
import os
import numpy as np
from ast import literal_eval

arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
             'FVNB','APC','NEB','PB',
             'UB','FPNB','ROTPVC']

if not os.path.exists("features_wavelet"):
    os.makedirs("feature_wavelet", exist_ok = True)


for i in range(len(arritmias)):

    muestra = len(os.listdir('./segmetados/'+str(arritmias[i])))
    k = i
    fe = {}
    print(arritmias[i])
    
    for j in range(muestra - 1):

        dp = pd.read_csv('./segmetados/'+ arritmias[i] +'/'+str(j + 1) +'.csv')
        
        pot = dp.get('0').values
        
        cA_coeff = []
        cD_coeff = []
        
        cA , cD = pywt.dwt(pot,'db8')
        cA_coeff.append(cA)
        cD_coeff.append(cD)
        
        for k in range(3):
            cA , cD = pywt.dwt(cA,'db8')
            cA_coeff.append(cA)
            cD_coeff.append(cD)
        
        
        fe[j] = np.concatenate([cA_coeff[3],cD_coeff[3],cD_coeff[2]])
    
    
    tam = muestra - 1
    
    if(tam > 0):
        
        feat = np.zeros((tam ,94))
        for w in range(muestra - 1):
            for x in range(94):
                feat[w][x] = fe[w][x]
        
        
        pca = PCA(n_components=18)
        pca.fit(feat)
        X = pca.transform(feat)
        
        df = pd.DataFrame(X)
        
        df.to_csv("feature_wavelet/"+arritmias[i]+".csv", index = False)






