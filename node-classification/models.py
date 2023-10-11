from typing import Union, Tuple

import torch
from torch import Tensor
from torch_geometric import nn
from torch_geometric.nn import GCN, GCNConv, GATConv, SAGEConv, GINConv, MLP, GATv2Conv, MessagePassing, DirGNNConv
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
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



class DirGraphSAGEModelWrapper(GraphSAGE):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers)
        self.out_channels = out_channels
        self.final_layer = SAGEConv(hidden_channels, out_channels)
        self.final_layer = DirGNNConv(self.final_layer, alpha=0.5, root_weight=True)

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> DirGNNConv:
        return DirGNNConv(SAGEConv(in_channels, out_channels, **kwargs), alpha=0.5, root_weight=True)

    def forward(self, x: torch.Tensor, edge_index: Adj):
        x = super().forward(x, edge_index)
        output = self.final_layer(x, edge_index)
        return output, x


class DirGATGNNConv(DirGNNConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, size: Size = None,
            return_attention_weights=None) -> Tensor:
        r""""""
        x_in = self.conv_in(x, edge_index, edge_attr=edge_attr, size=size, return_attention_weights=return_attention_weights)
        x_out = self.conv_out(x, edge_index.flip([0]), edge_attr=edge_attr, size=size, return_attention_weights=return_attention_weights)
        out = self.alpha * x_out + (1 - self.alpha) * x_in
        if self.root_weight:
            out = out + self.lin(x)
        return out


class DirGATModelWrapper(GAT):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, v2: bool):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers, v2=v2)
        self.out_channels = out_channels
        self.final_layer = GATConv(hidden_channels, out_channels, heads=8)
        self.final_layer = DirGATGNNConv(self.final_layer, alpha=0.5, root_weight=False)

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> DirGATGNNConv:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return DirGATGNNConv(Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs), alpha=0.5, root_weight=False)

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_attr: OptTensor=None):
        x = super().forward(x, edge_index, edge_attr=edge_attr)
        output = self.final_layer(x, edge_index, edge_attr=edge_attr)
        return output, x


class DirGCNGNNConv(DirGNNConv):
    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        r""""""
        x_in = self.conv_in(x, edge_index, edge_weight=edge_weight)
        x_out = self.conv_out(x, edge_index.flip([0]), edge_weight=edge_weight)
        out = self.alpha * x_out + (1 - self.alpha) * x_in
        if self.root_weight:
            out = out + self.lin(x)
        return out

class DirGCNModelWrapper(GCN):

    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        # Create the model to extract the node embeddings then pass these through a linear layer for classification
        super().__init__(in_channels, hidden_channels, num_layers)
        self.out_channels = out_channels
        self.final_layer = GCNConv(hidden_channels, out_channels)
        self.final_layer = DirGCNGNNConv(self.final_layer, alpha=0.5, root_weight=False)

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> DirGCNGNNConv:
        return DirGCNGNNConv(GCNConv(in_channels, out_channels, **kwargs), alpha=0.5, root_weight=False)

    def forward(self, x: torch.Tensor, edge_index: Adj, edge_weight: OptTensor=None):
        x = super().forward(x, edge_index, edge_weight=edge_weight)
        output = self.final_layer(x, edge_index, edge_weight=edge_weight)
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
    elif model_name == 'DirGraphSAGE':
        return DirGraphSAGEModelWrapper(in_channels, hidden_channels, num_layers, out_channels)
    elif model_name == 'DirGATv1':
        return DirGATModelWrapper(in_channels, hidden_channels, num_layers, out_channels, False)
    elif model_name == 'DirGCN':
        return DirGCNModelWrapper(in_channels, hidden_channels, num_layers, out_channels)
    else:
        raise ValueError(f'Invalid model name: {model_name}')
