import torch
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_sparse import SparseTensor


def load_cora():
    cora_dataset = Planetoid("/tmp/cora", name="cora", split="full")
    cora_data = cora_dataset[0]
    cora_data
    # Get the edge indices and node features for our model. General set up variables for running with all the models
    edge_indices = cora_data.edge_index
    node_features = cora_data.x
    neighbour_dataset = cora_data

    # Get masks and training labels for each split
    train_mask = cora_data.train_mask
    train_y = cora_data.y[train_mask]
    valid_mask = cora_data.val_mask
    valid_y = cora_data.y[valid_mask]
    test_mask = cora_data.test_mask
    test_y = cora_data.y[test_mask]

    num_classes = 7

    return node_features, num_classes, edge_indices, train_y, train_mask, valid_y, valid_mask, test_y, test_mask


def load_cora_sparse():
    cora_dataset = Planetoid("/tmp/cora", name="cora", split="full")
    cora_data = cora_dataset[0]
    # Get the edge indices and node features for our model. General set up variables for running with all the models
    edge_indices = cora_data.edge_index
    sparse_tensor_adj = SparseTensor.from_edge_index(edge_indices)
    node_features = cora_data.x

    return sparse_tensor_adj, node_features

def load_ogbn_arxiv():
    dataset = PygNodePropPredDataset(name="ogbn-arxiv")

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    arxiv_data = dataset[0]
    arxiv_data.y = arxiv_data.y.squeeze()
    arxiv_data.node_year = arxiv_data.node_year.squeeze()

    node_features = arxiv_data.x
    edge_indices = arxiv_data.edge_index

    # Get masks and training labels for each split
    train_mask = train_idx
    train_y = arxiv_data.y[train_mask]
    valid_mask = valid_idx
    valid_y = arxiv_data.y[valid_mask]
    test_mask = test_idx
    test_y = arxiv_data.y[test_mask]

    num_classes = 40

    return node_features, num_classes, edge_indices, train_y, train_mask, valid_y, valid_mask, test_y, test_mask

def load_ogbn_arxiv_sparse():
    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
    node_features = data.x
    return data.adj_t, node_features

def load_ogbl_citation2_sparse():
    dataset = PygLinkPropPredDataset(
        name='ogbl-citation2', transform=T.ToSparseTensor())
    data = dataset[0]
    node_features = data.x
    return data.adj_t, node_features


def load_dataset(dataset_name):
    if dataset_name == "Cora":
        return load_cora()
    elif dataset_name == "ogbn-arxiv":
        return load_ogbn_arxiv()


def load_dataset_sparse(dataset_name):
    if dataset_name == "Cora":
        return load_cora_sparse()
    elif dataset_name == "ogbn-arxiv":
        return load_ogbn_arxiv_sparse()
    elif dataset_name == "ogbl-citation2":
        return load_ogbl_citation2_sparse()

