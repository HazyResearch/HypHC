"""Triplet sampling utils."""

import numpy as np
from tqdm import tqdm


def samples_triples(n_nodes, num_samples):
    num_samples = int(num_samples)
    all_nodes = np.arange(n_nodes)
    mesh = np.array(np.meshgrid(all_nodes, all_nodes))
    pairs = mesh.T.reshape(-1, 2)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    n_pairs = pairs.shape[0]
    if num_samples < n_pairs:
        print("Generating all pairs subset")
        subset = np.random.choice(np.arange(n_pairs), num_samples, replace=False)
        pairs = pairs[subset]
    else:
        print("Generating all pairs superset")
        k_base = int(num_samples / n_pairs)
        k_rem = num_samples - (k_base * n_pairs)
        subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
        pairs_rem = pairs[subset]
        pairs_base = np.repeat(np.expand_dims(pairs, 0), k_base, axis=0).reshape((-1, 2))
        pairs = np.concatenate([pairs_base, pairs_rem], axis=0)
    num_samples = pairs.shape[0]
    triples = np.concatenate(
        [pairs, np.random.randint(n_nodes, size=(num_samples, 1))],
        axis=1
    )
    return triples


def generate_all_triples(n_nodes):
    triples = []
    for n1 in tqdm(np.arange(n_nodes)):
        for n2 in np.arange(n1 + 1, n_nodes):
            for n3 in np.arange(n2 + 1, n_nodes):
                triples += [(n1, n2, n3)]
    return np.array(triples)
