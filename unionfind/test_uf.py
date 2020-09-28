import numpy as np
import unionfind

if __name__ == '__main__':
    uf = unionfind.UnionFind(5)
    uf.merge(np.array([[0, 1], [2, 3], [0, 4], [3, 4]]))
    print(uf.parent)
    print(uf.tree)
