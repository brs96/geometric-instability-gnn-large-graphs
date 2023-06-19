import argparse
import os

import torch
# from dataset_loader import load_cora, load_dataset, load_dataset_sparse
import sys
sys.path.append('../node-classification')
from ggi import GGI
from utils.dataset_loader import load_dataset_sparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("In lp_metrics.py main")

    # CHANGE these to match you directory naming structure
    # Loop over the files in the directory
    model_directories = []
    directory = f'model_results/{MODEL}_{DATASET}/'
    model_directories.append(directory)


    for model_dir in model_directories:
        model_directory = model_dir
        output_path = model_directory + "analysis"

        if args.metric == "gram":
            # Load the dataset
            sparse_adj, node_features = load_dataset_sparse(DATASET)

            NUM_NODES = node_features.shape[0]
            print("NUM_NODES: ", NUM_NODES)

            ggi = GGI(DATASET, MODEL, device, NUM_NODES)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            ggi.ggi_over_dataset(sparse_adj, device, model_directory, output_path)
