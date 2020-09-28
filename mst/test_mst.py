import numpy as np
import mst

if __name__ == '__main__':
    x = np.array([0, 1, 3, 7, 15], dtype=np.float)
    dists = np.abs(x[np.newaxis, :] - x[:, np.newaxis])
    print(dists)
    print(mst.mst(dists, 5))
    print(-dists)
    print(mst.mst(-dists, 5))

    A = np.arange(16, dtype=np.float).reshape((4, 4))
    print(A)
    B = mst.reorder(A, np.array([3, 2, 1, 0]), 4)
    print(B)
