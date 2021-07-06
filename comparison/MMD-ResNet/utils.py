# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 00:16:44 2019

@author: urixs
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch
from torch.autograd import Variable
    
def compute_dist_mat(X, Y=None, device=torch.device("cpu")):
    """
    Computes nxm matrix of squared distances
    args:
        X: nxd tensor of data points
        Y: mxd tensor of data points (optional)
    """
    if Y is None:
        Y = X
       
    X = X.to(device=device)    
    Y = Y.to(device=device)  
    dtype = X.data.type()
    dist_mat = Variable(torch.Tensor(X.size()[0], Y.size()[0]).type(dtype)).to(device=device) 

    for i, row in enumerate(X.split(1)):
        r_v = row.expand_as(Y)
        sq_dist = torch.sum((r_v - Y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat

def nn_search(X, Y=None, k=10):
    """
    Computes nearest neighbors in Y for points in X
    args:
        X: nxd tensor of query points
        Y: mxd tensor of data points (optional)
        k: number of neighbors
    """
    if Y is None:
        Y = X
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(Y)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids
    
def compute_scale(Dis, k=5):
    """
    Computes scale as the max distance to the k neighbor
    args:
        Dis: nxk' numpy array of distances (output of nn_search)
        k: number of neighbors
    """
    scale = np.median(Dis[:, k - 1])
    return scale

def compute_kernel_mat(D, scale, device=torch.device('cpu')):
     """
     Computes RBF kernal matrix
     args:
        D: nxn tenosr of squared distances
        scale: standard dev 
     """
     W = torch.exp(-D / (scale ** 2))

     return W 
