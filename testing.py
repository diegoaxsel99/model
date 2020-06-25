# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:15:29 2020

@author: Matador
"""

from import_data import get, plot_corr, visualizacion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
"""
primera prueba

X_train, X_test, y_train, y_test = get("standardscaler", random = 2)

knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train,y_train)

print(knn.score(X_test,y_test))
"""


# segunda prueba

# ds = pd.read_csv('matriz/stats_features.csv')
# dw = pd.read_csv('matriz/coeff_features.csv')
# dx = pd.read_csv('matriz/time_features.csv')
# ds.drop(['label'], axis='columns', inplace=True)

# dn = pd.concat([ds,dx], axis = 1)
# dn = pd.concat([dn,dw], axis = 1)
# [f, c] = dn.shape
# c = c - 1

# data  = np.zeros((f,c))
# target = dn.get('label').values
# columns = dn.columns

# for i in range(len(columns) - 1):
    
#     data[:,i] = dn.get(columns[i]).values
knn = KNeighborsClassifier(n_neighbors = 5)

X_train, X_test, y_train, y_test  = get('cross', 2,pca = True)

knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))

X_train, X_test, y_train, y_test  = get('cross', 2,pca = False)

knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))


"""
test 3

X_train, X_test, y_train, y_test = get('cross', random = 2)

iris_dataframe = pd.DataFrame(X_train)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8)

"""
# data = get('Dataframe' , random = 2)
# plot_corr(data)

