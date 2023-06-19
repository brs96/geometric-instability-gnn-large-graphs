import csv
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import block_diag

####### PLOTS #######

def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
      epoch_stats: dict containg metrics about one epoch
      training_stats: dict containing lists of metrics along training
    Returns:
      updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key, val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats


def plot_stats(training_stats, figsize=(5, 5), name=""):
    """ Create one plot for each metric stored in training_stats
    """
    stats_names = [key[6:] for key in training_stats.keys() if key.startswith('train_')]
    f, ax = plt.subplots(len(stats_names), 1, figsize=figsize)
    if len(stats_names) == 1:
        ax = np.array([ax])
    for key, axx in zip(stats_names, ax.reshape(-1, )):
        axx.plot(
            training_stats['epoch'],
            training_stats[f'train_{key}'],
            label=f"Training {key}")
        axx.plot(
            training_stats['epoch'],
            training_stats[f'val_{key}'],
            label=f"Validation {key}")
        axx.set_xlabel("Training epoch")
        axx.set_ylabel(key)
        axx.legend()
    plt.title(name)


def get_color_coded_str(i, color):
    return "\033[3{}m{}\033[0m".format(int(color), int(i))


def print_color_numpy(map, list_graphs):
    """ print matrix map in color according to list_graphs
    """
    list_blocks = []
    for i, graph in enumerate(list_graphs):
        block_i = (i + 1) * np.ones((graph.num_nodes, graph.num_nodes))
        list_blocks += [block_i]
    block_color = block_diag(*list_blocks)

    map_modified = np.vectorize(get_color_coded_str)(map, block_color)
    print("\n".join([" ".join(["{}"] * map.shape[0])] * map.shape[1]).format(
        *[x for y in map_modified.tolist() for x in y]))


def saving_model_metadata(seed, model_name, hidden_dim, epochs, num_layers, file_path):
    with open(file_path + str(seed) + "_model_metadata.txt", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seed", "model_name", "hidden_dim", "epochs", "num_layers"])
        writer.writerow([seed, model_name, hidden_dim, epochs, num_layers])


def save_training_info(training_stats: dict, node_embedding: torch.Tensor, file_path: str, filename: str):
  with open(file_path + filename + "_train_stats.pkl", 'wb') as fp:
    pickle.dump(training_stats, fp)
    print('Training stats saved successfully to file: ' + filename)
  torch.save(node_embedding, file_path + filename + "_emb.pt")
  print('Node embedding saved successfully to file: ' + filename)

def save_predictions(predictions: torch.Tensor,file_path: str, prediction_filename: str):
    print(f'Predictions number is : {predictions.shape}')
    torch.save(predictions, file_path + prediction_filename + ".pt")
    print('Test prediction saved successfully to file: ' + prediction_filename)


def load_training_info(file_path: str, filename: str):
  with open(file_path + filename + ".pkl", 'rb') as fp:
    train_stats = pickle.load(fp)
    print('Training stats successfully loaded from file: ' + filename)
  node_embedding = torch.load(file_path + filename + "_emb.pt")
  print('Node embedding successfully loaded from file: ' + filename)
  return train_stats, node_embedding


def save_final_results(final_results: List, file_path: str, seed, filename: str):
  # write training data info to a file
  with open(file_path + str(seed) + filename + ".pkl", 'ab') as fp:
    pickle.dump(final_results, fp)
    print('Final results saved successfully to file: ' + filename)

def load_final_results(file_path: str, filename: str):
  with open(file_path + filename + ".pkl", 'rb') as fp:
    print('Final results found in file: ' + filename)
    while True:
      try:
        yield pickle.load(fp)
      except EOFError:
        break