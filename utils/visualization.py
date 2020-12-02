"""Visualization utils."""
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.lca import hyp_lca


def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    """Plots geodesic between x and y."""
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=1.5, alpha=1)


def complete_tree(tree, leaves_embeddings):
    """Get embeddings of internal nodes from leaves' embeddings using LCA construction."""

    def _complete_tree(embeddings, node):
        children = list(tree.neighbors(node))
        if len(children) == 2:
            left_c, right_c = children
            left_leaf = is_leaf(tree, left_c)
            right_leaf = is_leaf(tree, right_c)
            if left_leaf and right_leaf:
                pass
            elif left_leaf and not right_leaf:
                embeddings = _complete_tree(embeddings, right_c)
            elif right_leaf and not left_leaf:
                embeddings = _complete_tree(embeddings, left_c)
            else:
                embeddings = _complete_tree(embeddings, right_c)
                embeddings = _complete_tree(embeddings, left_c)
            embeddings[node] = hyp_lca_numpy(embeddings[left_c], embeddings[right_c])
        return embeddings

    n = leaves_embeddings.shape[0]
    tree_embeddings = np.zeros((2 * n - 1, 2))
    tree_embeddings[:n, :] = leaves_embeddings
    root = max(list(tree.nodes()))
    tree_embeddings = _complete_tree(tree_embeddings, root)
    return tree_embeddings


def hyp_lca_numpy(x, y):
    """Computes the hyperbolic LCA in numpy."""
    x = torch.from_numpy(x).view((1, 2))
    y = torch.from_numpy(y).view((1, 2))
    lca = hyp_lca(x, y, return_coord=True)
    return lca.view((2,)).numpy()


def is_leaf(tree, node):
    """check if node is a leaf in tree."""
    return len(list(tree.neighbors(node))) == 0


def plot_tree_from_leaves(ax, tree, leaves_embeddings, labels, color_seed=1234):
    """Plots a tree on leaves embeddings using the LCA construction."""
    circle = plt.Circle((0, 0), 1.0, color='r', alpha=0.1)
    ax.add_artist(circle)
    n = leaves_embeddings.shape[0]
    embeddings = complete_tree(tree, leaves_embeddings)
    colors = get_colors(labels, color_seed)
    ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=50, alpha=0.6)

    for n1, n2 in tree.edges():
        x1 = embeddings[n1]
        x2 = embeddings[n2]
        plot_geodesic(x1, x2, ax)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    return ax


def get_colors(y, color_seed=1234):
    """random color assignment for label classes."""
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return [colors[k] for k in y]
