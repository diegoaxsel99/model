# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:20:44 2020

@author: Matador
"""

from import_data import get
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data ,target = get('data',2)
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = 0 ,test_size = 0.2)


svm = SVC(degree = 5).fit(X_train, y_train)
y_pred = svm.predict(X_test)

score = []
matrix_c = confusion_matrix(y_test, y_pred)
score.append(svm.score(X_test,y_test))
