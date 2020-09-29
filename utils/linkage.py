"""Decoding utils."""

import time

import numpy as np
import torch
from tqdm import tqdm

from mst import mst
from unionfind import unionfind
from utils.lca import hyp_lca


### Single linkage using MST trick

# @profile
def sl_np_mst(similarities):
    n = similarities.shape[0]
    ij, _ = mst.mst(similarities, n)
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree

def sl_from_embeddings(xs, S):
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    sim_mat = S(xs0, xs1)  # (n, n)
    return sl_np_mst(sim_mat.numpy())

### Single linkage using naive union find

# @profile
def nn_merge_uf_fast_np(xs, S, partition_ratio=None, verbose=False):
    """ Uses Cython union find and numpy sorting

    partition_ratio: either None, or real number > 1
    similarities will be partitioned into buckets of geometrically increasing size
    """
    n = xs.shape[0]
    # Construct distance matrix (negative similarity; since numpy only has increasing sorting)
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    dist_mat = -S(xs0, xs1)  # (n, n)
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))

    # Keep only unique pairs (upper triangular indices)
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat = dist_mat[idx]

    # Sort pairs
    if partition_ratio is None:
        idx = np.argsort(dist_mat, axis=0)
    else:
        k, ks = ij.shape[0], []
        while k > 0:
            k = int(k // partition_ratio)
            ks.append(k)
        ks = np.array(ks)[::-1]
        if verbose:
            print(ks)
        idx = np.argpartition(dist_mat, ks, axis=0)
    ij = ij[idx]

    # Union find merging
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree
