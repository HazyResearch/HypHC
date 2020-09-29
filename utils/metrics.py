"""Evaluation utils."""

import numpy as np

#from mst import reorder
from mst import mst
from utils.tree import descendants_traversal, descendants_count


def dasgupta_cost_iterative(tree, similarities):
    """ Non-recursive version of DC. Also works on non-binary trees """
    n = len(list(tree.nodes()))
    root = n - 1

    cost = [0] * n

    desc = [None] * n  # intermediate computation: children of node

    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) == 0:
                desc[node] = [node]

            else:
                # Intermediate computations
                desc[node] = [d for c in children_ for d in desc[c]]

                # Cost at this node
                # cost_ = similarities[desc[node]].T[desc[node]].sum()
                # cost_ -= sum([similarities[desc[c]].T[desc[c]].sum() for c in children_])
                # cost_ = cost_ / 2.0
                # This is much faster for imbalanced trees
                cost_ = sum([similarities[desc[c0]].T[desc[c1]].sum() for i, c0 in enumerate(children_) for c1 in
                             children_[i + 1:]])
                cost_ *= len(desc[node])

                cost[node] = cost_ + sum([cost[c] for c in children_])  # recursive cost

                # Free intermediate computations (otherwise, up to n^2 space for recursive descendants)
                for c in children_:
                    desc[c] = None

            assert node == stack.pop()
    return 2 * cost[root]


def dasgupta_cost(tree, similarities):
    """ Non-recursive version of DC for binary trees.

    Optimized for speed by reordering similarity matrix for locality
    """
    n = len(list(tree.nodes()))
    root = n - 1
    n_leaves = len(similarities)

    leaves = descendants_traversal(tree)
    n_desc, left_desc = descendants_count(tree)

    cost = [0] * n  # local cost for every node

    # reorder similarity matrix for locality
    # similarities = similarities[leaves].T[leaves] # this is the bottleneck; is there a faster way?
    similarities = mst.reorder(similarities, np.array(leaves), n_leaves)  # this is the bottleneck; is there a faster way?

    # Recursive computation
    children = [list(tree.neighbors(node)) for node in range(n)]  # children remaining to process
    stack = [root]
    while len(stack) > 0:
        node = stack[-1]
        if len(children[node]) > 0:
            stack.append(children[node].pop())
        else:
            children_ = list(tree.neighbors(node))

            if len(children_) < 2:
                pass
            elif len(children_) == 2:
                left_c = children_[0]
                right_c = children_[1]

                left_range = [left_desc[left_c], left_desc[left_c] + n_desc[left_c]]
                right_range = [left_desc[right_c], left_desc[right_c] + n_desc[right_c]]
                cost_ = np.add.reduceat(
                    np.add.reduceat(
                        similarities[
                        left_range[0]:left_range[1],
                        right_range[0]:right_range[1]
                        ], [0], axis=1
                    ), [0], axis=0
                )
                cost[node] = cost_[0, 0]

            else:
                assert False, "tree must be binary"
            assert node == stack.pop()

    return 2 * sum(np.array(cost) * np.array(n_desc))
