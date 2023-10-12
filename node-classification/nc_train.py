import argparse
import os
import random
import sys
from time import time

import numpy as np
import torch

from models import create_model
from training import train_eval_loop_gnn
from log_embeddings_and_predictions import saving_model_metadata
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from utils.dataset_loader import load_dataset

seeds = [4193977854, 1863727779, 170173784, 2342954646, 116846604, 2105922959, 2739899259, 1024258131, 806299656,
         880019963,
         1818027900, 2135956485, 3710910636, 1517964140, 4083009686, 2455059856, 400225693, 89475662, 361232447,
         3647665043,
         1221215631, 2036056847, 1860537279, 516507873, 3692371949, 3300171104, 2794978777, 3303475786, 2952735006,
         572297925]

# seeds = [4193977854, 1863727779, 170173784, 2342954646, 116846604, 2105922959, 2739899259, 1024258131, 806299656,
#          880019963]


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_with_seed(MODEL, DATASET, seed, hidden_dim, epochs, num_layers,
                    node_features, num_classes, edge_indices, train_y, train_mask, valid_y, valid_mask, test_y,
                    test_mask,
                    file_path):
    t = time()
    set_seeds(seed)
    model = create_model(model_name=MODEL, in_channels=node_features.shape[-1], hidden_channels=hidden_dim,
                         num_layers=num_layers,
                         out_channels=num_classes)
    model = model.to(device)
    train_stats = train_eval_loop_gnn(MODEL, DATASET, model, epochs, edge_indices, node_features, train_y, train_mask,
                                      node_features, valid_y, valid_mask, node_features, test_y, test_mask,
                                      num_classes, seed, file_path, device)

    print(train_stats)
    return train_stats


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in seeds:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str)
        parser.add_argument('--model', type=str)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--num_layers', type=int, default=3)
        args = parser.parse_args()

        set_seeds(seed)

        DATASET = args.dataset
        MODEL = args.model

        file_path = f'model_results/{MODEL}_{DATASET}/'
        # create folder if it does not exist already
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        saving_model_metadata(seed, MODEL, args.hidden_dim, args.epochs, args.num_layers, file_path)

        node_features, num_classes, edge_indices, train_y, train_mask, valid_y, valid_mask, test_y, test_mask = load_dataset(
            DATASET)
        train_with_seed(MODEL, DATASET, seed, args.hidden_dim, args.epochs, args.num_layers,
                        node_features, num_classes, edge_indices, train_y, train_mask, valid_y, valid_mask, test_y,
                        test_mask,
                        file_path)

