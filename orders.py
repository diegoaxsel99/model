# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:45:53 2020

@author: Matador
"""
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import scipy.signal as sign
from import_data import get, plot_corr,visualizacion
import pandas as pd
import os

#%% mostrar el filtrado de la señal

#path = os.path.join('midb/','101')
file = '100'
[signal , info] = wfdb.rdsamp('mitdb/' + file, sampto = 1800)

time = np.linspace(0,5,len(signal))

variables = {}
variables['raw'] = signal[:,0]
variables['fs'] = 360

fc1 = 35
fc2 = 30

wp = 2 *(40 / variables['fs'])
ws = 2 *(55 / variables['fs']) 
gpass = 1
gstop = 40

[a,b] = sign.iirdesign(wp, ws, gpass, gstop, ftype='butter')
variables['filt1'] = sign.filtfilt(a,b,variables['raw'])

variables['filt2'] = sign.medfilt(variables['filt1'], 5)

names = ['raw','filt1','filt2']
titles = ['señal gruda', 
          'filtro irr pasa banda (0.5 Hz - 60 Hz)',
          'filtro medfilt orden 5']

[fig , ax] = plt.subplots(3,1)

fig.suptitle("tratamiento de la señal", fontsize = 25)

for i in range(3):
    
    axes = ax[i]
    axes.set_title(titles[i])
    axes.plot(time,variables[names[i]])
    axes.grid()
    axes.set_ylabel('voltage (mv)')

axes.set_xlabel('tiempo (seg)')
if not os.path.exists("info/orders"):
    os.makedirs("info/orders", exist_ok = True)

fig.savefig('info/orders/tratamiento_de_la_señal.png')

#%% generar la matriz de correlacion

data = get('Dataframe' , random = 2)
plot_corr(data)

plt.savefig('info/orders/matriz_correlacion.png')
plt.close('all')

#%% generar grafica de dispersion

X_train, X_test, y_train, y_test = get('cross', random = 2)

iris_dataframe = pd.DataFrame(X_train)

grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8)

plt.savefig('info/orders/diagrama de dispersion.png')

#%% aplicacion de pca

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# mismo numero de componentes

pca = PCA(n_components = 0.99)

data,target = get('data',random = 2)
data_rescaled = MinMaxScaler().fit_transform(data)

transformaciones ={}

transformaciones['mismo numero'] = pca.fit_transform(data)

df =pd.DataFrame(transformaciones['mismo numero'])

plot_corr(df)

plt.savefig('info/orders/matriz_correlacion_pca1.png')
plt.close('all')

# visualizacion
visualizacion(data,target)
plt.savefig('info/orders/visualizacion.png')

arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
              'APC','NEB','PB','UB','FPNB','ROTPVC']

ntarget = []

for i in range(len(target)):
    print(target[i])
    
    ntarget.append(arritmias[target[i]])

dt = pd.DataFrame(ntarget, columns = ['target'])
dfinal = pd.concat([df,dt],axis = 1)

dfinal.to_csv('matriz/pca_features.csv', index =False)
