# cython: boundscheck=False, wraparound=False, cdivision=True
""" MST algorithm. Adapted from scipy.cluster._hierarchy.pyx """

cimport cython
# from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np
# ctypedef np.int_t DTYPE_t
# cdef bint boolean_variable = True

cdef extern from "numpy/npy_math.h":
    cdef enum:
        NPY_INFINITYF

def mst(double[:,:] dists, int n):
    """ MAXIMUM Spanning Tree of a dense adjacency matrix using Prim's Algorithm
    Parameters
    ----------
    dists : ndarray
        A condensed matrix stores the pairwise distances of the observations.
    n : int
        The number of observations.
    Returns
    -------
    Z : ndarray[int64],   shape (n-1, 2)
    d : ndarray[float64], shape (n-1, 1)
        Edges in the MST in sorted order; indices, lengths
    """

    ij = np.empty((n - 1, 2), dtype=np.int)
    cdef long[:, :] Z = ij
    l = np.empty(n-1)
    cdef double[:] l_ = l

    # Which nodes were already merged.
    cdef int[:] merged = np.zeros(n, dtype=np.intc)

    # Best distance of node i to current tree
    cdef double[:] D = np.empty(n)
    D[:] = -NPY_INFINITYF
    cdef int[:] j = np.empty(n, dtype=np.intc)

    cdef int i, k
    cdef int x = 0 # The node just added to the tree
    cdef int y # stores the next candidate node to add
    cdef double dist, current_max

    # x = 0
    for k in range(n - 1):
        merged[x] = 1
        current_max = -NPY_INFINITYF
        for i in range(n):
            if merged[i] == 1:
                continue

            dist = dists[x,i]
            if D[i] < dist:
                D[i] = dist
                j[i] = x

            if current_max < D[i]:
                current_max = D[i]
                y = i
            # print(x, i, current_max)

        # for linkage, this works if you assign it x instead, but the proof is subtle
        Z[k, 0] = j[y]
        # Z[k, 0] = x
        Z[k, 1] = y
        # Z[k, 2] = current_min
        l_[k] = current_max
        x = y

    # Sort Z by distances
    order = np.argsort(l, kind='mergesort')[::-1]
    ij = ij[order]
    l = l[order]

    # # Find correct cluster labels and compute cluster sizes inplace.
    # label(ij, n)

    return ij, l

def reorder(double[:,:] A, long[:] idx, int n):
    """
    A : (n, n)
    idx: (n)
    """
    B = np.empty((n, n))
    cdef double[:,:] B_ = B
    cdef int i, j, k
    cdef double[:] row
    for i in range(n):
        k = idx[i]
        # row = A[k]
        for j in range(n):
            B_[i, j] = A[k,idx[j]]
            # B_[i, j] = row[idx[j]]
    return B
