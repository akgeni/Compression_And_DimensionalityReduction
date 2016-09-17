# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:21:52 2016

@author: akgeni
"""

from __future__ import division
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sb  
from scipy.io import loadmat  

%matplotlib inline

data_path = '''C:\\Users\\akgeni\\Documents\\MachineLearning\\
MachineLearning_Andrew\\ExercisesInPython\\data'''

data = loadmat(data_path + '\\ex7data1.mat')
X = data['X']
print(X)

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X[:, 0], X[:, 1])

def pca(X):
    
    # normalize the features
    X = (X - np.mean(X)) / np.std(X)
    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    
    return U, S, V

U, S, V = pca(X)

print(U, S, V)

def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)
    
Z = project_data(X, U, 1)

def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)

print(Z)

fig, ax = plt.subplots(figsize=(10, 8))
recovered_data = recover_data(Z, U, 1)
ax.scatter(recovered_data[:, 0], recovered_data[:, 1])

faces = loadmat(data_path + "\\ex7faces.mat")
X = faces['X']
X.shape

sample_face = np.reshape(X[3,:], (32, 32))
plt.imshow(sample_face)

U, S, V = pca(X)
Z = project_data(X, U, 100)
X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3, :], (32, 32))
plt.imshow(face)

