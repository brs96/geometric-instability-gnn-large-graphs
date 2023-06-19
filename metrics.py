import argparse
import os

import torch
from metric_calc import run_metric_calculations
from utils.dataset_loader import load_dataset, load_dataset_sparse
from ggi import GGI

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    DATASET = args.dataset
    MODEL = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CHANGE these to match you directory naming structure
    # Loop over the files in the directory
    model_directories = []
    directory = f'model_results/{MODEL}_{DATASET}/'
    model_directories.append(directory)

    for model_dir in model_directories:
        model_directory = model_dir
        output_path = model_directory + "analysis"

        if args.metric == "jaccard":
            node_features, _, _, _, _, _, _, _, _ = load_dataset(DATASET)

            NUM_NODES = node_features.shape[0]
            print("NUM_NODES: ", NUM_NODES)

            run_metric_calculations(embedding_dir=model_directory, results_dir=output_path, tests=[args.metric],
                                    num_nodes=NUM_NODES, num_processes=4)
        if args.metric == "2ndcos":
            node_features, _, _, _, _, _, _, _, _ = load_dataset(DATASET)

            NUM_NODES = node_features.shape[0]
            print("NUM_NODES: ", NUM_NODES)

            run_metric_calculations(embedding_dir=model_directory, results_dir=output_path, tests=[args.metric],
                                    num_nodes=NUM_NODES, num_processes=4, load_knn=True)
        if args.metric == "gram":
            sparse_adj, node_features = load_dataset_sparse(DATASET)

            NUM_NODES = node_features.shape[0]

            ggi = GGI(DATASET, MODEL, device, NUM_NODES)
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            ggi.ggi_over_dataset(sparse_adj, device, model_directory, output_path)
