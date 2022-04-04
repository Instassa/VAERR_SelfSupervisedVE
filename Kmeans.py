import time
import torch
from matplotlib import pyplot as plt
from pykeops.torch import LazyTensor

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

def KMeans(x, K=20, n_init=1, centroids = []):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    B, N, D = x.shape  # Number of batches, samples, dimension of the ambient space
    # print(B, N, D)

    x = x.contiguous().type(torch.float)
    if centroids == []:
        c = x[0, :K, :].clone()  # Simplistic initialization for the centroids
    else:
        c = centroids
    
    # x = x.contiguous().type('float32')
    x_i = LazyTensor(x.view(N*B, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(n_init):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        # a = (x_i - c_j) ** 2  # (N, K) symbolic squared distances
        # D_ij = torch.dot(a.view(-1), torch.ones_like(a).view(-1))
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        # print('c:', c.shape, 'cl:', cl[:, None].repeat(1, D).shape, 'x:', x.shape)
        c.scatter_add_(0, cl[:, None].repeat(1, D), x.view(N*B, D))

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c #nearest cluster per point, centroids