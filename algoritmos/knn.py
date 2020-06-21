# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:28:00 2020

@author: Matador
"""


import os 

cwd = os.getcwd()
cwd = cwd[0:len(cwd) - 11]
os.chdir(cwd)

from import_data import get, coeff
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

#%% mostrar el rendimiento del sistema 
if not os.path.exists('resultados/knn'):
    os.makedirs('resultados/knn')
data, target = get('data' ,random  = 2)
for i in range(3):

    random = i 
    X_train, X_test, y_train, y_test = get("cross",random)
    k_range = range(1,13,2)
    acc = []
    acc1 = []
    print(i)
    sens = []
    pre =[]
    for j in k_range:
        
        knn = KNeighborsClassifier(n_neighbors = j)
        knn.fit(X_train,y_train)
        
        y_pred = knn.predict(X_test)
        y_pred1 = knn.predict(X_train)
        
        acc.append(metrics.accuracy_score(y_pred,y_test))
        acc1.append(metrics.accuracy_score(y_pred1,y_train))
        auxs,auxp = coeff(X_test, y_test, knn)
        
        sens.append(auxs)
        pre.append(auxp)
    
    plt.figure()
    plt.title("resultado generados con knn con random state de " + str(random))
    plt.xlabel("# vecinos")
    plt.ylabel('accurency')
    plt.plot(k_range,acc)
    plt.plot(k_range,acc1)
    plt.legend(["test","training"])
    plt.show()
    
    
    plt.savefig('resultados/knn/knn_resul '+ str(random) +' .png')
    
    plt.figure()
    plt.plot(k_range,sens)
    plt.plot(k_range,pre)
    plt.title('sensibilidad y especifidad con random ' +str(random))
    plt.xlabel('# vecinos')
    plt.show()
    plt.legend(['sensibilidad','especificidad'])
    plt.savefig('resultados/knn/knn_seesp '+ str(random) +' .png')
    
    plt.close('all')