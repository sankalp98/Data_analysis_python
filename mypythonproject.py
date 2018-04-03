#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 00:43:57 2018

@author: Apple
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

#X_set, Y_set = X, Y

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Kernel SVC(to make our model learn the data of 300 users)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

#Predicting the results of 100 users based on what our model learnt earlier on the 300 users
Y_pred = classifier.predict(X_test)

#Making the Confusion Matrix( This matrix helps to know the number of correct and wrong predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

#Visualising the 300 set Results
from matplotlib.colors import ListedColormap
X_fin, Y_fin = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_fin[:, 0].min() - 1, stop = X_fin[:, 0].max() + 1, step = 0.01), np.arange(start = X_fin[:, 1].min() - 1, stop = X_fin[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_fin)):
    plt.scatter(X_fin[Y_fin == j, 0], X_fin[Y_fin == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVC')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualising the 100 set Results
from matplotlib.colors import ListedColormap
X_fin, Y_fin = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_fin[:, 0].min() - 1, stop = X_fin[:, 0].max() + 1, step = 0.01), np.arange(start = X_fin[:, 1].min() - 1, stop = X_fin[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_fin)):
    plt.scatter(X_fin[Y_fin == j, 0], X_fin[Y_fin == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVC')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
















