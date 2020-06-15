# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:43:19 2020

@author: Matador
"""

from import_data import get, coeff
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np

if not os.path.exists('resultados/knn'):
    os.makedirs('resultados/knn')

for i in range(3):

    data,target = get()
    random = i
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=random ,test_size = 0.2)
    
    k_range = range(1,31)
    acc = []
    print(i)
    sens = []
    pre =[]
    for i in k_range:
        
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train,y_train)
        acc.append(knn.score(X_test, y_test))
        auxs,auxp = coeff(X_test, y_test, knn)
        
        sens.append(auxs)
        pre.append(auxp)
    
    plt.figure()
    plt.title("resultado generados con knn con random state de " + str(random))
    plt.xlabel("# vecinos")
    plt.ylabel('accurency')
    plt.plot(k_range,acc)
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
    
# y_pred = knn.predict(X_test)

# matrix_c = confusion_matrix(y_test, y_pred)
# diag = np.diag(matrix_c)
# f = matrix_c[:,0]
# c = matrix_c[0,:] 


# tp = sum(diag) - diag[0]
# tn = diag[0]
# fp = sum(c) - c[0]
# fn = sum(f) - f[0]

# sens = tp /(tp + fn)
# esp =  tn /(tn + fp)
