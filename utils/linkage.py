"""Decoding utils."""

import time

import numpy as np
import torch
from tqdm import tqdm

from mst import mst
from unionfind import unionfind
from utils.lca import hyp_lca


# Union find based strategy

# Adapted from https://python-algorithms.readthedocs.io/en/stable/_modules/python_algorithms/basic/union_find.html
class UF:
    """An implementation of union find data structure.
    It uses weighted quick union by rank with path compression.
    """

    def __init__(self, N):
        """Initialize an empty union find object with N items.

        Args:
            N: Number of items in the union find object.
        """

        self._link = list(range(N))
        self._count = N
        self._rank = [0] * N

        # Variables that track the binary tree of merges
        self._next_id = N
        self._tree = [-1] * (2 * N - 1)  # parent links
        self._id = list(range(N))  # the map from UF trees to merge tree identifiers

    def find(self, p):
        """Find the set identifier for the item p."""

        link = self._link
        while p != link[p]:
            # Path compression using halving.
            q = link[link[p]]
            link[p] = q
            p = q
        return p

    def count(self):
        """Return the number of items."""

        return self._count

    def connected(self, p, q):
        """Check if the items p and q are on the same set or not."""

        return self.find(p) == self.find(q)

    def union(self, p, q):
        """Combine sets containing p and q into a single set."""

        link = self._link
        rank = self._rank

        i = self.find(p)
        j = self.find(q)
        if i == j:
            return False

        self._count -= 1
        if rank[i] < rank[j]:
            link[i] = j
            self._merge(j, i)
        elif rank[i] > rank[j]:
            link[j] = i
            self._merge(i, j)
        else:
            link[j] = i
            rank[i] += 1
            self._merge(i, j)
        return True

    def _merge(self, i, j):
        """ track the tree changes when node j gets merged into node i """
        self._tree[self._id[i]] = self._next_id
        self._tree[self._id[j]] = self._next_id
        self._id[i] = self._next_id
        self._next_id += 1

    def __str__(self):
        """String representation of the union find object."""
        return " ".join([str(x) for x in self._link])

    def __repr__(self):
        """Representation of the union find object."""
        return "UF(" + str(self) + ")"


# @profile
def nn_merge_uf(xs, D, debug=False, verbose=False):
    """ A version of the single linkage decoding in pure python """

    n = xs.shape[0]
    # Construct distance matrix
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    dist_mat = D(xs0, xs1)  # (n, n)
    i = np.tile(np.arange(n, dtype=int)[:, None], (1, n))  # can do with meshgrid?
    j = np.tile(np.arange(n, dtype=int), (n, 1))
    ij = np.stack([i.flatten(), j.flatten()], axis=-1)
    ij = ij[np.argsort(dist_mat.flatten(), axis=0)]
    if debug:
        print(ij)

    uf = UF(n)
    if verbose:
        for i, j in tqdm(ij):
            uf.union(i, j)
    else:
        for i, j in ij:
            uf.union(i, j)

    if debug:
        print(uf._tree)

    return uf._tree


# @profile
def sl_np_sort(similarities):
    n = similarities.shape[0]
    dist_mat = -similarities
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))

    # Keep only unique pairs (upper triangular indices)
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat = dist_mat[idx]

    # Sort pairs
    print("Sorting similarities...", flush=True)
    idx = np.argsort(dist_mat, axis=0)
    ij = ij[idx]

    # Union find merging
    print("Merging...", flush=True)
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree


# @profile
def sl_np_mst(similarities):
    n = similarities.shape[0]
    # dist_mat = -similarities
    ij, _ = mst.mst(similarities, n)
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree


# sl = sl_np_sort
sl = sl_np_mst

def sl_from_embeddings(xs, S):
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    sim_mat = S(xs0, xs1)  # (n, n)
    return sl(sim_mat.numpy())


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
    # i = np.tile(np.arange(n, dtype=int)[:, None], (1, n)) # can do with meshgrid?
    # j = np.tile(np.arange(n, dtype=int), (n, 1))
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))

    # Keep only unique pairs (upper triangular indices)
    # ij = np.stack([i.flatten(), j.flatten()], axis=-1)
    # idx = ij[:,0]<ij[:,1]
    # ij = ij[idx]
    # dist_mat = dist_mat.flatten()[idx]
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
        # breakpoint()
    ij = ij[idx]

    # Union find merging
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree


# @profile
def nn_merge_uf_fast(xs, S):
    """ Pytorch version using similarity functions

    xs: embeddings of shape (n, d)

    On GPU, can only handle up to ~20k nodes before saturating 16GB memory
    """
    n = xs.shape[0]
    # Construct distance matrix
    xs0 = xs[None, :, :]  # (1, n, d)
    xs1 = xs[:, None, :]  # (n, 1, d)
    dist_mat = S(xs0, xs1)  # (n, n)
    i, j = torch.meshgrid(torch.arange(n, dtype=int), torch.arange(n, dtype=int))
    ij = torch.stack([i, j], axis=-1).reshape(-1, 2)
    idx = torch.argsort(dist_mat.view(-1), descending=True)
    ij = ij[idx]
    ij = ij.numpy()
    uf = unionfind.UnionFind(n)
    uf.merge(ij)
    return uf.tree


def test_merge_uf(n=100, gpu=False, cpu=False, numpy=False, python=False, linkage=False, compare=True, **np_args):
    d = 2
    xs = np.random.normal(size=(n, d))
    xs = torch.from_numpy(xs)
    xs = xs / xs.norm(p=2, dim=-1, keepdim=True) * .5

    tree = None

    if gpu:
        print("Testing union find tree Cython torch gpu")
        start = time.perf_counter()
        tree_ = nn_merge_uf_fast(xs.to('cuda'), lambda x, y: torch.sum(x * y, dim=-1))
        end = time.perf_counter()
        print(end - start, "seconds")

        if compare:
            print("Results agree:", tree == tree_)
            tree = tree_

    if cpu:
        print("Testing union find tree Cython torch cpu")
        start = time.perf_counter()
        tree_ = nn_merge_uf_fast(xs, lambda x, y: torch.sum(x * y, dim=-1))
        end = time.perf_counter()
        print(end - start, "seconds")

        if compare:
            print("Results agree:", tree == tree_)
            tree = tree_

    if numpy:
        print("Testing union find tree Cython numpy")
        start = time.perf_counter()
        tree_ = nn_merge_uf_fast_np(xs, lambda x, y: torch.sum(x * y, dim=-1), **np_args)
        end = time.perf_counter()
        print(end - start, "seconds")

        if compare:
            print("Results agree:", tree == tree_)
            tree = tree_

    if python:
        print("Testing union find tree")
        start = time.perf_counter()
        dist_fn = lambda x, y: -hyp_lca(x, y, return_coord=False)
        tree_ = nn_merge_uf(xs, dist_fn, verbose=True)
        end = time.perf_counter()
        print(end - start, "seconds")

        if compare:
            print("Results agree:", tree == tree_)
            tree = tree_

    if linkage:
        print("Testing union find using single linkage MST algorithm")
        start = time.perf_counter()
        tree = sl_from_embeddings(xs, lambda x, y: torch.sum(x * y, dim=-1))
        end = time.perf_counter()
        print(end - start, "seconds")

        if compare:
            print("Results agree:", tree == tree_)
            tree = tree_


if __name__ == '__main__':
    # test_merge_uf(n=1000, numpy=True, linkage=True, compare=True, partition_ratio=1.2)
    test_merge_uf(n=1000, numpy=True, linkage=True, compare=True)
