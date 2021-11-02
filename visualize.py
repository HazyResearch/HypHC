"""Script to visualize the HypHC clustering."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch

from datasets.loading import load_data
from model.hyphc import HypHC
from utils.poincare import project
from utils.visualization import plot_tree_from_leaves

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Hierarchical Clustering.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="path to a directory with a torch model_{seed}.pkl and a config.json files saved by train.py."
                        )
    parser.add_argument("--seed", type=str, default=0, help="model seed to use")
    args = parser.parse_args()

    # load dataset
    config = json.load(open(os.path.join(args.model_dir, "config.json")))
    config_args = argparse.Namespace(**config)
    _, y_true, similarities = load_data(config_args.dataset, config_args.large_dataset)

    # build HypHC model
    model = HypHC(similarities.shape[0], config_args.rank, config_args.temperature, config_args.init_size,
                  config_args.max_scale)
    params = torch.load(os.path.join(args.model_dir, f"model_{args.seed}.pkl"), map_location=torch.device('cpu'))
    model.load_state_dict(params, strict=False)
    model.eval()

    # decode tree
    tree = model.decode_tree(fast_decoding=True)
    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)
    leaves_embeddings = project(leaves_embeddings).detach().cpu().numpy()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax = plot_tree_from_leaves(ax, tree, leaves_embeddings, labels=y_true)
    fig.savefig(os.path.join(args.model_dir, f"embeddings_{args.seed}.png"))
