import torch
from torch_geometric import nn
from torch_geometric.nn import GCN, GCNConv, GATConv, SAGEConv, GINConv, MLP, GATv2Conv
from torch_geometric.typing import Adj
from torch_geometric.nn import GIN, GAT, GraphSAGE


class GCNModelWrapper(GCN):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        # use one less layer as our final graph layer can downsize for us
        # super().__init__(in_channels, hidden_channels, num_layers-1)
        super().__init__(in_channels, hidden_channels, num_layers)
        self.out_channels = out_channels
        self.final_layer = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x = super().forward(x, edge_index)
        output = self.final_layer(x, edge_index)
        return output, x


class GATModelWrapper(GAT):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, v2: bool):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers, v2=v2)
        self.out_channels = out_channels
        if v2:
            self.final_layer = GATv2Conv(hidden_channels, out_channels, heads=8)
        else:
            self.final_layer = GATConv(hidden_channels, out_channels, heads=8)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x = super().forward(x, edge_index)
        output = self.final_layer(x, edge_index)
        return output, x


class GraphSAGEModelWrapper(GraphSAGE):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers)
        self.out_channels = out_channels
        self.final_layer = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x = super().forward(x, edge_index)
        output = self.final_layer(x, edge_index)
        return output, x


class GINWrapper(GIN):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers)
        self.out_channels = out_channels
        mlp = MLP(
            [hidden_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        self.final_layer = GINConv(mlp, out_channels)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x = super().forward(x, edge_index)
        output = self.final_layer(x, edge_index)
        return output, x


def create_model(model_name: str, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
    if model_name == 'GCN':
        return GCNModelWrapper(in_channels, hidden_channels, num_layers, out_channels)
    elif model_name == 'GATv1':
        return GATModelWrapper(in_channels, hidden_channels, num_layers, out_channels, False)
    elif model_name == 'GATv2':
        return GATModelWrapper(in_channels, hidden_channels, num_layers, out_channels, True)
    elif model_name == 'GraphSAGE':
        return GraphSAGEModelWrapper(in_channels, hidden_channels, num_layers, out_channels)
    elif model_name == 'GIN':
        return GINWrapper(in_channels, hidden_channels, num_layers, out_channels)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
