# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:34:05 2020

@author: Matador
"""

import pywt
import pandas as pd
from sklearn.decomposition import FastICA, PCA
import os
import numpy as np
from ast import literal_eval
from scipy.signal import find_peaks,peak_prominences
from scipy import stats

def dwt():
    
    """
    extrae las caracteristicas de los coeffientes wavelet
    """
    
    
    arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
                 'APC','NEB','PB',
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
            
            
            pca = PCA(n_components = 18)
            pca.fit(feat)
            X = pca.transform(feat)
            
            df = pd.DataFrame(X)
            
            df.to_csv("feature_wavelet/"+arritmias[i]+".csv", index = False)
            
def coeff():
    
    dwt()
    
    if not os.path.exists("matriz"):
        os.makedirs("matriz", exist_ok = True)
    
    arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
                  'APC','NEB','PB','UB','FPNB']
    
    base = {}
    data = []
    label = []
    
    for k in range(len(arritmias)):
    
        ds = pd.read_csv('feature_wavelet/'+arritmias[k]+'.csv')
        
        aux0 = len(ds.get('0').values)
        [f,c] = ds.shape
        
        dat = np.zeros((aux0, c))
        
        
        for j in range(c):
            
            aux = ds.get(str(j)).values
            
            for i in range(len(aux)):
                
                dat[i][j] = aux[i]
                
        data.append(dat)
    
    
    base['label'] = np.asarray(label)
    
    tam = 0
    
    for i in range(len(data)):
        
        tam = tam + len(data[i])
        
    dataf = np.zeros([tam,c])
    
    conc = 0 
    conf = 0
    
    for i in range(len(data)):
        
        [f,c] = data[i].shape
        
        for j in range(f):
            conc = 0
            for k in range(c):
                
                dataf[conf,conc] = data[i][j,k]
                conc = conc + 1
                
                if(k == c - 1):
                    conf = conf + 1
                    label.append(i)
                    
    columns = []
    base['label'] = label
               
    for i in range (c):
        
        base['c'+str(i)] = dataf[:,i]
        columns.append('c'+str(i))
    
    columns.append('label')
    dv = pd.DataFrame(base , columns = columns)
    dv.to_csv('matriz/coeff_features.csv' , index = False)
    
def time():
    
    
    def diff(fx,t):
        
        h = (max(t) - min(t)) / len(t)
        
        dfx = []
        dfx.append((fx[1] - fx[0])/h)
        
        for i in range(len(t) - 1):
            dfx.append((fx[i+1] - fx[i]) / h)
            
        return np.asarray(dfx)
    
    arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
                 'APC','NEB','PB','UB','FPNB','ROTPVC']
    
    if not os.path.exists("matriz"):
        os.makedirs("matriz", exist_ok = True)
    
    p4 = []
    for i in range(len(arritmias)):
    
        muestra = len(os.listdir('./segmetados/'+str(arritmias[i])))
        print(arritmias[i])
        ia = i   
        for j in range(muestra - 1):
            
            file = 'segmetados/' + arritmias[i] + '/'+ str(j + 1) + '.csv'
            signal = pd.read_csv(file).get('0').values

            t = np.linspace(0,1,int(0.6*360))
            
            peaks, _ = find_peaks(signal, distance = 30)

            prominences = peak_prominences(signal, peaks)[0]
            contour_heights = signal[peaks] - prominences
            
            peaks2, _ =find_peaks(-signal, distance =30)
            prominences2 = peak_prominences(-signal, peaks2)[0]
            contour_heights = signal[peaks] - prominences
            
            
            
            p1 = np.sort(prominences)[len(prominences)-2:len(prominences)]        
            p2 = np.sort(prominences2)[len(prominences2)-2:len(prominences2)]
            
            pp1 = []
            pp2 = []
            p3 = []
            for i in range(2):
                
                if(len(p2) == 2):
                    pp2.append(peaks2[np.where(prominences2 == p2[i])[0][0]])
                else:
                    pp2.append(peaks2[np.where(prominences2 == p2[0])[0][0]])
                    
                
                if(len(p1) == 2):
                     pp1.append(peaks[np.where(prominences == p1[i] )[0][0]])
                else:
                     pp1.append(peaks[np.where(prominences == p1[0] )[0][0]])
                     
                
            for i in range(4):
                if(i<2):
                    p3.append(pp1[i]/360)
                else:
                    p3.append(pp2[i - 2]/360)
            
            p3 = np.sort(p3)
            p4.append(p3)
            
            i = ia
    p4 = np.asarray(p4)
    
    columns = ['t1','t2','t3','t4']
    base ={}
    for q in range(len(columns)):
        base[columns[q]] = p4[:,q] 
    
    df = pd.DataFrame(base, columns = columns)
    df.to_csv('matriz/time_features.csv' , index = False)

def Stats():
    
    if not os.path.exists("matriz"):
        os.makedirs("matriz", exist_ok = True)
    
    arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC'
                 ,'APC','NEB','PB',
                 'UB','FPNB','ROTPVC']
    
    mean = []
    var = []
    median = []
    std = []
    label = []
    for i in range(len(arritmias)):
        
        print(arritmias[i])
        muestra = len(os.listdir('./segmetados/'+str(arritmias[i])))
        
        for j in range(muestra - 1):
            
            dp = pd.read_csv('./segmetados/'+ arritmias[i] +'/'+str(j + 1) +'.csv')
            pot = dp.get('0').values
            
            
            mean.append(np.mean(pot))
            var.append(np.var(pot))
            median.append(np.median(pot))
            
            std.append(np.std(pot))
            
            if (i >= 6):
                
                label.append(i - 1)
            else:
                label.append(i)
    
    columns = ["mean","var","median","std","label"]
    
    base = { "mean": mean ,"var":var,"median":median,"std": std,"label":label}
    
    dv = pd.DataFrame(base, columns = columns)
    
    dv.to_csv("matriz/stats_features.csv", index = False)

def all_features():
    
    coeff()
    Stats()
    time()
    
    ds = pd.read_csv('matriz/stats_features.csv')
    dw = pd.read_csv('matriz/coeff_features.csv')
    dx = pd.read_csv('matriz/time_features.csv')
    ds.drop(['label'], axis='columns', inplace=True)
    
    dn = pd.concat([ds,dx], axis = 1)
    dn = pd.concat([dn,dw], axis = 1)

    dn.to_csv('matriz/all_features.csv', index = False)