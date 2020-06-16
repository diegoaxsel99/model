# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:43:19 2020

@author: Matador
"""

from import_data import get, coeff
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import numpy as np

#%% mostrar el rendimiento del sistema 

if not os.path.exists('resultados/knn'):
    os.makedirs('resultados/knn')

for i in range(3):

    random = i 
    X_train, X_test, y_train, y_test = get("cross",random)
    
    k_range = range(1,13,2)
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

#%% loo
loo = LeaveOneOut()

data,target = get("data",0)
knn = KNeighborsClassifier(n_neighbors = 3)
acc = []
i = 0

total = loo.get_n_splits(data)
for train_i,test_i in loo.split(data):
    
    print(str((i/total) * 100)+'%')
    i = i + 1
    X_train,X_test = data[train_i],data[test_i]
    y_train,y_test = target[train_i],target[test_i]
            
    knn.fit(X_train,y_train)
    acc.append(knn.score(X_test, y_test))
    
plt.figure()
plt.title("resultado generados con knn y leave one out " + str(random))
plt.xlabel("# iteraciones")
plt.ylabel('accurency')
plt.plot(acc)
plt.show()


plt.savefig('resultados/knn/knn_resul loo'+ str(random) +' .png')
plt.close('all') 

#%% kfold
knn = KNeighborsClassifier(n_neighbors = 3)
por = []
splits = range(2,16)

for i in splits:
    
    print(i)
    kf = KFold(n_splits = i)
    acc = []
    
    for train_i, test_i in kf.split(data):
        
        X_train, X_test = data[train_i], data[test_i]
        y_train, y_test = target[train_i], target[test_i]
        
        knn.fit(X_train, y_train)
        
        acc.append(knn.score(X_test, y_test))
    
    por.append(sum(acc) / len(acc))

plt.figure()
plt.title("resultado generados con knn y kfold " + str(random))
plt.xlabel("# splits")
plt.ylabel('accurency')
plt.plot(splits , por)
plt.show()

plt.savefig('resultados/knn/knn_kfold'+ str(random) +' .png')
plt.close('all') 
    



