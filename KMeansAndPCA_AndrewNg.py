# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:02:07 2016

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

raw_data = loadmat(data_path + '\\ex7data2.mat')
X = raw_data['X']


def find_closest_cluster(X, intial_centroids):
    m = X.shape[0]
    k = intial_centroids.shape[0]
    idx = np.zeros(m)
    
    for i in range(m):
        closest_dist = 999999.
        closest_centroid = None
        for j in range(k):
            dist_ij = np.sum((X[i,:] - intial_centroids[j, :]) ** 2)
            if dist_ij < closest_dist:
                closest_dist = dist_ij                
                idx[i] = j
    return idx

intial_centroids = np.array([[3, 3], [6, 6], [8, 5]])
idx = find_closest_cluster(X, intial_centroids)
idx[:3]

def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids
    

compute_centroids(X, idx, 3)


def run_k_means(X, initial_centroids, max_iters):  
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids
    
def init_centroids(X, k):  
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids



idx, centroids = run_k_means(X, intial_centroids, 10)  


cluster1 = X[np.where(idx == 0)[0],:]  
cluster2 = X[np.where(idx == 1)[0],:]  
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')  
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')  
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')  
ax.legend()  

# compress image using KMeans, 
# concept is that centroids are representative of colors within given cluster
# now we need less number of bits to represent a pixel

image_data = loadmat(data_path + '\\bird_small.mat')
print(image_data)
print(image_data['A'].shape)
plt.imshow(image_data['A'])
plt.imsave(data_path + '\\original.png', image_data['A'])

# normalize the values
im_data = image_data['A'] / 255
X = np.reshape(im_data, (im_data.shape[0] * im_data.shape[1], im_data.shape[2]))

initial_centroids = init_centroids(X, 16)

idx, centroids = run_k_means(X, initial_centroids, 10)
idx = find_closest_cluster(X, initial_centroids)
X_recovered = centroids[idx.astype(int), :]
X_recovered = np.reshape(X_recovered, (im_data.shape[0], im_data.shape[1], im_data.shape[2]))
plt.imshow(X_recovered)
plt.imsave(data_path + '\\compressed.png', X_recovered)





