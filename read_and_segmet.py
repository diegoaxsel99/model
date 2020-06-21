# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:46:23 2020

"""
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sign
from wfdb import processing
import pandas as pd
import os 
#%% leer y mostar la señal
"""
toma la base de datos y los segmenta en la clase de arrimias con 
respecto a la variable limit

"""
def read_and_segment():
    
    if (os.path.exists('./segmetados')):
        
        contn = len(os.listdir('./segmetados/normal beat'))
        contl = len(os.listdir('./segmetados/LBBBB'))
        contr = len(os.listdir('./segmetados/RBBBB'))
        conta = len(os.listdir('./segmetados/AAPB'))
        contv = len(os.listdir('./segmetados/PVC'))
        contf = len(os.listdir('./segmetados/FVNB'))
        contJ = len(os.listdir('./segmetados/NPB'))
        contA = len(os.listdir('./segmetados/APC'))
        conts = len(os.listdir('./segmetados/POESB'))
        conte = len(os.listdir('./segmetados/VEB'))
        contj = len(os.listdir('./segmetados/NEB'))
        contz = len(os.listdir('./segmetados/PB'))
        contq = len(os.listdir('./segmetados/UB'))
        contf = len(os.listdir('./segmetados/FPNB'))
        contr = len(os.listdir('./segmetados/ROTPVC'))
        
    else:
        contn = 0
        contl = 0
        contr = 0
        conta = 0
        contv = 0
        contF = 0
        contJ = 0
        contA = 0
        conts = 0
        conte = 0
        contj = 0
        contz = 0
        contq = 0
        contf = 0
        contr = 0
    
    limit = 3000    
    
    
    vector = [100,101,102,103,104,105,106,
              107,111,113,114,115,116,
              117,118,119,121,122,123,124,
              200,201,202,203,205,207,209,
              210,212,213,214,215,217,219,
              220,221,222,223,230,231,232,
              233,234]
    
    
    for j in vector:
    
        file = str(j)
        
        print("!!!!")
        print(j)
        print("!!!!")
        
        [signal , info] = wfdb.rdsamp('mitdb/' + file, sampto = 648000)
        
        variables = {}
        
        columns = ['MLII','V5','fs','time','sample','symbols',
                   'clean','label']
        
        variables[info['sig_name'][0]] = signal[:,0]
        variables[info['sig_name'][1]] = signal[:,1]
        variables['fs'] = info['fs']
        
        tf = info['sig_len'] / variables['fs']
        
        variables['time'] = np.linspace(0,tf,info['sig_len'])
        
        record = wfdb.rdrecord('mitdb/' + file, sampto = 648000)
        ann = wfdb.rdann('mitdb/' + file, 'atr', sampto = 648000)
        
        variables['sample'] = ann.sample
        variables['symbols'] = ann.symbol
        
        variables['sample'] = variables['sample'].tolist()
        
        # wfdb.plot_wfdb(record=record, 
        #                     annotation = ann,
        #                     plot_sym=True,
        #                     time_units ='seconds', 
        #                     title='MIT-BIH Record' + file,
        #                     figsize=(10,4), 
        #                     ecg_grids='all')
        
        #%% filtrado de la señal
        
        fc1 = 35
        fc2 = 30
        
        wp = 2 *(40 / variables['fs'])
        ws = 2 *(55 / variables['fs']) 
        gpass = 1
        gstop = 40
        
        [a,b] = sign.iirdesign(wp, ws, gpass, gstop, ftype='butter')
        variables['clean'] = sign.filtfilt(a,b,
                                                variables[info['sig_name'][0]])
        
        variables['clean'] = sign.medfilt(variables['clean'], 5)
        
        #%% qrs detection y segmetacion
        
        variables['qrs_inds'] = processing.xqrs_detect(variables['clean'], 
                                                       fs = variables['fs'])
        
        dirs = ['normal beat','LBBBB','RBBBB','AAPB','PVC','FVNB','NPB',
                'APC','POESB','VEB','NEB','PB','UB','FPNB','ROTPVC']
        
        folder = 'segmetados/'
        
        for i in range(len(dirs)):
            
            if not os.path.exists(folder + dirs[i]):
                os.makedirs(folder + dirs[i], exist_ok = True)
        
        for i in range(len(variables['qrs_inds']) - 1):
        
            
            print(i)
        
            punto = variables['qrs_inds'][i]
            ref = variables
            
            tipo = variables['symbols'][i + 1]
            
            signals = variables['clean'][punto - 72 : punto + 144]
            
            df = pd.DataFrame(signals)
            
            
            if(tipo == 'N' and contn < limit):
                
                contn = contn + 1
                df.to_csv(folder + 'Normal beat/'+ str(contn)+'.csv', 
                          index = False)
            
            if(tipo == 'L' and contl < limit):
                contl = contl + 1
                df.to_csv(folder + 'LBBBB/'+ str(contl)+'.csv', 
                          index = False)
                
            if(tipo == 'R' and contr < limit):
                contr = contr + 1
                df.to_csv(folder + 'RBBBB/'+ str(contr)+'.csv', 
                          index = False)
            
            if(tipo == 'a' and conta < limit):
                conta = conta + 1
                df.to_csv(folder + 'AAPB/'+ str(conta)+'.csv', 
                          index = False)
                
            if(tipo == 'V' and contv < limit):
                contv = contv + 1
                df.to_csv(folder + 'PVC/'+ str(contv)+'.csv', 
                          index = False)
                
            if(tipo == 'F' and contf < limit):
                contF = contF + 1
                df.to_csv(folder + 'FVNB/'+ str(contf)+'.csv', 
                          index = False)
                
            if(tipo == 'J' and contJ < limit):
                contJ = contJ + 1
                df.to_csv(folder + 'NPB/'+ str(contJ)+'.csv', 
                          index = False)
            
            if(tipo == 'A' and contA < limit):
                contA = contA + 1
                df.to_csv(folder + 'APC/'+ str(contA)+'.csv', 
                          index = False)
                
            if(tipo == 'S' and conts < limit):
                conts = conts + 1
                df.to_csv(folder + 'POESB/'+ str(conts)+'.csv', 
                          index = False)
            
            if(tipo == 'E' and conte < limit):
                conte = conte + 1
                df.to_csv(folder + 'VEB/'+ str(conte)+'.csv', 
                          index = False)
            
            if(tipo == 'j' and contj < limit):
                contj = contj + 1
                df.to_csv(folder + 'NEB/'+ str(contj)+'.csv', 
                          index = False)
                
            if(tipo == '/' and contz < limit):
                contz = contz + 1
                df.to_csv(folder + 'PB/'+ str(contz)+'.csv', 
                          index = False)
            
            if(tipo == 'Q' and contq < limit):
                contq = contq + 1
                df.to_csv(folder + 'UB/'+ str(contq)+'.csv', 
                          index = False)
                
            if(tipo == 'f' and contf < limit):
                contf = contf + 1
                df.to_csv(folder + 'FPNB/'+ str(contf)+'.csv', 
                          index = False)
            
            if(tipo == 'r' and contr < limit):
                contr = contr + 1
                df.to_csv(folder + 'ROTPVC/'+ str(contr)+'.csv', 
                          index = False)
            
    
    
    
        
        
        
        
        
    
