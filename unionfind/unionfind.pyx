""" Union find data structure. Adapted from https://github.com/eldridgejm/unionfind """

cimport cython
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

ctypedef np.int_t DTYPE_t
cdef bint boolean_variable = True

cdef class UnionFind:
    cdef int n
    cdef int *parent
    cdef int *rank
    # cdef int _n_sets

    # Variables that track the binary tree of merges
    cdef int _next_id
    cdef int *_tree  # parent links
    cdef int *_id  # the map from UF trees to merge tree identifiers

    def __cinit__(self, int n):
        self.n = n
        self.parent = <int *> malloc(n * sizeof(int))
        self.rank = <int *> malloc(n * sizeof(int))

        cdef int i
        for i in range(n):
            self.parent[i] = i

        # self._n_sets = n

        self._next_id = n
        self._tree = <int *> malloc((2 * n - 1) * sizeof(int))
        for i in range(2 * n - 1):
            self._tree[i] = -1
        self._id = <int *> malloc(n * sizeof(int))
        for i in range(n):
            self._id[i] = i

    def __dealloc__(self):
        free(self.parent)
        free(self.rank)

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    cdef int _find(self, int i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self._find(self.parent[i])
            return self.parent[i]

    def find(self, int i):
        if (i < 0) or (i > self.n):
            raise ValueError("Out of bounds index.")
        return self._find(i)

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    # def union(self, int i, int j):
    cpdef bint union(self, int i, int j):
        # TODO probably best to split into internal cdef and external def with bounds checking
        # if (i < 0) or (i > self.n) or (j < 0) or (j > self.n):
        #     raise ValueError("Out of bounds index.")

        cdef int root_i, root_j
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i == root_j:
            return False
        else:
            # self._n_sets -= 1
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self._build(root_j, root_i)
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self._build(root_i, root_j)
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self._build(root_i, root_j)
            return True

    @cython.boundscheck(False)  # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    @cython.nonecheck(False)
    def merge(self, np.ndarray[DTYPE_t, ndim=2] ij):
        """ Merge a sequence of pairs """
        cdef int k
        for k in range(ij.shape[0]):
            self.union(ij[k, 0], ij[k, 1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef void _build(self, int i, int j):
        """ Track the tree changes when node j gets merged into node i """
        self._tree[self._id[i]] = self._next_id
        self._tree[self._id[j]] = self._next_id
        self._id[i] = self._next_id
        self._next_id += 1

    # property n_sets:
    #     def __get__(self):
    #         return self._n_sets
    @property
    def sets(self):
        return 2 * self.n - self._next_id

    @property
    def parent(self):
        return [self.parent[i] for i in range(self.n)]

    @property
    def tree(self):
        return [self._tree[i] for i in range(2 * self.n - 1)]
