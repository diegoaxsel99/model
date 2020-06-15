# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:21:24 2020

@author: Matador
"""

# import pandas as pd
# import matplotlib.pyplot as plt
# import pywt
# import numpy as np

# def plot_wavelet(ax, time2, signal, scales, waveletname = 'cmor', 
#                  cmap =plt.cm.seismic, title = '', ylabel = '', xlabel = ''):
#     dt=time2
#     coefficients, frequencies = pywt.cwt(signal, scales, waveletname, dt)

#     power = (abs(coefficients)) ** 2
#     period = frequencies
#     levels = [0.015625,0.03125,0.0625, 0.125, 0.25, 0.5, 1]
#     contourlevels = np.log2(levels) #original
#     time=range(2048)

#     im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)


#     ax.set_title(title, fontsize=20)
#     ax.set_ylabel(ylabel, fontsize=18)
#     ax.set_xlabel(xlabel, fontsize=18)
#     yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))    
#     ax.set_yticks(np.log2(yticks)) #original
#     ax.set_yticklabels(yticks) #original
#     ax.invert_yaxis()
#     ylim = ax.get_ylim()

#     cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
#     fig.colorbar(im, cax=cbar_ax, orientation="vertical")

#     return yticks, ylim



# df = pd.read_csv ('./segmetados/PB/30.csv')

# signal = df.get('0').values

# xrange=list(range(216))
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(xrange,signal, color="b", alpha=0.5, label='original signal')
# rec = lowpassfilter(signal, 0.4)
# ax.plot(xrange,rec, 'k', label='DWT smoothing}', linewidth=2)
# ax.legend()
# ax.set_title('Removing High Frequency Noise with DWT', fontsize=18)
# ax.set_ylabel('Signal Amplitude', fontsize=16)
# ax.set_xlabel('Sample No', fontsize=16)
# plt.margins(0)
# plt.show()


# scale_range = np.arange(2, 50) # number of scales
# fig, ax = plt.subplots(figsize=(12, 8))
# plot_wavelet(ax=ax, time2=sp, signal=signal, scales=scale_range,waveletname='cmor1.5-1.0',
#              title = "CWT of Signal")
# plt.show()


# df = pd.read_csv ('./segmetados/PB/30.csv')

# d1 = df.get('0').values

# cA, cD = pywt.dwt( d1, 'db1')

# coeff = []
# coeff.append(cD)

# for i in range(3):
    
#     cA, cD = pywt.dwt( cA, 'db1')
#     coeff.append(cD)
    

# plt.figure()
# plt.plot(d1)

# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.arange(512)
# y = np.sin(2*np.pi*x/32)
# coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
# plt.matshow(coef) # doctest: +SKIP
# plt.show() # doctest: +SKIP

# import pywt
# import numpy as np
# import matplotlib.pyplot as plt
# t = np.linspace(-1, 1, 200, endpoint=False)
# sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
# widths = np.arange(1, 31)
# cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
#             vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
# plt.show() # doctest: +SKIP



# import pandas as pd
# import matplotlib.pyplot as plt
# import pywt
# import numpy as np

# df = pd.read_csv ('./segmetados/PB/30.csv')

# pd = df.get('0').values


# fs = 360
# sampling_period = 1/fs

# coef, freqs = pywt.cwt(pd ,np.arange(1,30),'gaus1',sampling_period = sampling_period)

# plt.matshow(coef) # doctest: +SKIP
# plt.colorbar()
# plt.show() 

import pywt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import chirp
from mpl_toolkits.mplot3d import Axes3D
import os

wavelet = "mexh"

dire = os.getcwd()

xlabel = 'tiempo (segs)'
ylabel1 = 'amplitud (mV)'
ylabel2 = 'frecuencia (Hz)'

arritmias = ['normal beat','LBBBB','RBBBB','AAPB','PVC',
             'FVNB','NPB','APC','POESB','VEB','NEB','PB',
             'UB','FPNB','ROTPVC']

for i in range(len(arritmias)):

    muestra = os.listdir('./segmetados/'+str(arritmias[i]))
    
    print(i)
    
    dire_ima = './segmetados/'+str(arritmias[i])+'/'+'imagenes'
    
    if(len(muestra) != 0):
    
        fs = 360
        sampling_period = 1/fs
        
        if not os.path.exists(dire_ima):
            os.makedirs(dire_ima)
        
        df = pd.read_csv ('./segmetados/' + str(arritmias[i])+ '/2.csv')
        
        sd = df.get('0').values
        plt.figure()
        
        t = np.linspace(0,60,216)
        
        plt.subplot(1,2,1)
        plt.plot(t,sd)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel1)
        plt.title(arritmias[i] + str(' señal'))
        
        coef, freqs = pywt.cwt(sd, np.arange(1, 50), 
                               wavelet,
                                sampling_period=sampling_period)
        
        dt = 1/360  # 100 Hz sampling
        freqs = pywt.scale2frequency(wavelet, range(1,50)) / dt
        
        
        #plt.figure(figsize=(5, 2))
        plt.subplot(1,2,2)
        plt.pcolor(t, freqs, coef)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel2)
        plt.colorbar()
        plt.ylim([0,50])
        plt.title(arritmias[i] +str(wavelet))
        
        X, Y = np.meshgrid( t, freqs)
        plt.savefig(dire_ima + '/figura 1 ' + str(arritmias[i]) + str(wavelet))
        
        fig = plt.figure()
        ax = plt.axes (projection = '3d')
        ax.set_ylabel('Frecuencia', fontsize=16)
        ax.set_xlabel('tiempo', fontsize=16)
        ax.set_ylim([0 ,50])
        surf = ax.plot_surface(X,Y,coef, cmap = 'viridis')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig(dire_ima + '/figura 2 '+str(arritmias[i])+ " "+ str(wavelet))
        
        plt.close('all')

