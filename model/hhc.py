"""Hyperbolic hierarchical clustering model."""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lca import hyp_lca
from utils.linkage import nn_merge_uf_fast_np
from utils.metrics import dasgupta_cost
from utils.poincare import project


class HHC(nn.Module):
    """
    Hyperbolic embedding model for hierarchical clustering.
    """

    def __init__(self, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, margin=0.0, max_scale=1. - 1e-3,
                 n_classes=1):
        super(HHC, self).__init__()
        self.n_nodes = n_nodes
        self.embeddings = nn.Embedding(n_nodes, rank)
        self.temperature = temperature
        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin)
        self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        self.embeddings.weight.data = project(
            self.scale * (2 * torch.rand((n_nodes, rank)) - 1.0)
        )
        self.init_size = init_size
        self.max_scale = max_scale

    def anneal_temperature(self, anneal_factor):
        """

        @param anneal_factor: scalar for temperature decay
        @type anneal_factor: float
        @return:
        @rtype:
        """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """

        @type embeddings: Tensor
        @param embeddings:
        @return:
        @rtype:
        """
        min_scale = 1e-2  # self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    def loss(self, triple_ids, similarities):
        """Computes the HypHC loss.
        Args:
            triple_ids: B x 3 tensor with triple ids
            similarities: B x 3 tensor with pairwise similarities for triples 
                          [s12, s13, s23]
        """
        e1 = self.embeddings(triple_ids[:, 0])
        e2 = self.embeddings(triple_ids[:, 1])
        e3 = self.embeddings(triple_ids[:, 2])
        e1 = self.normalize_embeddings(e1)
        e2 = self.normalize_embeddings(e2)
        e3 = self.normalize_embeddings(e3)
        d_12 = hyp_lca(e1, e2, return_coord=False)
        d_13 = hyp_lca(e1, e3, return_coord=False)
        d_23 = hyp_lca(e2, e3, return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / self.temperature, dim=-1)
        w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
        total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)

    def decode_tree_apx(self, leaves_embeddings, similarities, return_tree=False):

        def greedy_angle_cut(cluster_ids, first_split):
            theta_cluster = theta[cluster_ids]  # should be sorted
            diff = theta_cluster[1:] - theta_cluster[:-1]
            diff = np.minimum(diff, 2 * np.pi - diff)
            if first_split:
                # Adding angle between first and last point
                theta_last_first = theta_cluster[-1] - theta_cluster[0]
                theta_last_first = min(theta_last_first, 2 * np.pi - theta_last_first)
                diff = np.concatenate([diff, np.array([theta_last_first])])
                i1, i2 = np.argpartition(-diff, 2)[:2]
                i1, i2 = min(i1, i2), max(i1, i2)
                left_ids = np.concatenate([cluster_ids[:i1 + 1], cluster_ids[i2 + 1:]])
                right_ids = cluster_ids[i1 + 1:i2 + 1]
            else:
                i = np.argmax(diff)
                left_ids, right_ids = cluster_ids[:i + 1], cluster_ids[i + 1:]
            return left_ids, right_ids

        def _top_down_cluster(ids, first_split, current, node_counter, tree, cost):
            if len(ids) > 1:
                left_ids, right_ids = greedy_angle_cut(ids, first_split)
                assert len(left_ids) + len(right_ids) == len(ids)
                assert len(left_ids) > 0
                assert len(right_ids) > 0
                if len(left_ids) > 1:
                    node_counter -= 1
                    left_idx = node_counter
                else:
                    left_idx = left_ids[0]
                if len(right_ids) > 1:
                    node_counter -= 1
                    right_idx = node_counter
                else:
                    right_idx = right_ids[0]

                # add parent child edges in tree and update cost
                tree.add_edge(current, left_idx)
                tree.add_edge(current, right_idx)
                cost += similarities[left_ids].T[right_ids].sum() * (len(left_ids) + len(right_ids))

                # recurse on subtrees
                tree, node_counter, cost = _top_down_cluster(left_ids, first_split=False, current=left_idx, tree=tree,
                                                             node_counter=node_counter, cost=cost)
                tree, node_counter, cost = _top_down_cluster(right_ids, first_split=False, current=right_idx, tree=tree,
                                                             node_counter=node_counter, cost=cost)

            elif len(ids) == 1:
                # reach leaf node stop here
                pass

            else:
                # this should not be reached
                raise NotImplementedError

            return tree, node_counter, cost

        x = leaves_embeddings[:, 0]
        y = leaves_embeddings[:, 1]
        z = x + 1j * y
        theta = np.angle(z)
        sort_ids = np.argsort(theta)
        theta = theta[sort_ids]
        similarities = similarities[sort_ids, :][:, sort_ids]
        tree = nx.DiGraph()
        n_nodes = leaves_embeddings.shape[0]
        root = 2 * n_nodes - 2
        tree.add_node(root)
        node_counter = root
        tree, node_counter, cost = _top_down_cluster(np.arange(n_nodes), first_split=True, current=root, tree=tree,
                                                     node_counter=node_counter, cost=0.0)
        cost *= 2
        if return_tree:
            mapping = dict(zip(range(len(sort_ids)), sort_ids))
            tree = nx.relabel_nodes(tree, mapping)
            return tree, cost
        else:
            return cost

    def decode_tree_uf(self, leaves_embeddings, similarities, **kwargs):
        """ Assume points are normalized to same radius """
        sim_fn = lambda x, y: torch.sum(x * y, dim=-1)
        parents = nn_merge_uf_fast_np(leaves_embeddings, S=sim_fn, **kwargs)
        tree = nx.DiGraph()
        for i, j in enumerate(parents[:-1]):
            tree.add_edge(j, i)
        return dasgupta_cost(tree, similarities)

    def decode_tree(self, similarities, fast_decoding):
        """Build a binary tree (nx graph) from leaves' embeddings."""
        leaves_embeddings = self.normalize_embeddings(self.embeddings.weight.data)
        leaves_embeddings = project(leaves_embeddings)
        if fast_decoding:
            # Greedy angle decoding
            leaves_embeddings = leaves_embeddings.detach().cpu().numpy()
            cost = self.decode_tree_apx(leaves_embeddings, similarities)
            # Unionfind decoding
            # leaves_embeddings = leaves_embeddings.detach().cpu()
            # cost = self.decode_tree_uf(leaves_embeddings, similarities, partition_ratio=1.2)
        else:
            cost = self.decode_tree_uf(leaves_embeddings, similarities)
        return cost
