import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, ModuleList, Sequential, ReLU
from torch_geometric.nn import TransformerConv, TopKPooling, GINEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool

torch.manual_seed(42)


class GINE(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)

        self.lin1 = Linear(dim_h * 4, dim_h * 4)
        self.lin2 = Linear(dim_h * 4, n_data_points)

        self.global_proj = Linear(1, dim_h + 1)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings

        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h2, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3, h4), dim=1)
        # h = torch.cat((h1, h2), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        return torch.nn.Softplus()(h)


class GINEGLOBAL(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(GINEGLOBAL, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)

        self.lin1 = Linear(dim_h * 4 + additional_feature_size, dim_h * 32)
        self.lin3 = Linear(dim_h * 32, dim_h * 16)
        self.lin2 = Linear(dim_h * 16, n_data_points)

        # self.global_proj = Linear(1, dim_h + 1)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings

        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        h = torch.cat((h1, h2, h3, h4, skip_features), dim=1)
        # h = torch.cat((h1, h2, h3, skip_features), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.lin3(h)
        h = h.relu()
        h = F.dropout(h, p=0.4, training=self.training)
        h = self.lin2(h)

        return torch.nn.Softplus()(h)


class ModelPredNumPeak(torch.nn.Module):

    def __init__(self,
                 node_feature_size,
                 edge_feature_size,
                 dim_h,
                 n_data_points,
                 additional_feature_size=12):
        super(ModelPredNumPeak, self).__init__()
        self.conv1 = GINEConv(
            Sequential(Linear(node_feature_size, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv2 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size
        )
        self.conv3 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.conv4 = GINEConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()),
            edge_dim=edge_feature_size)
        self.lin1 = Linear(dim_h * 4, dim_h * 4)
        self.lin2 = Linear(dim_h * 4, n_data_points)

    def forward(self, x, graph_level_feats, edge_attr, edge_index, batch_index):
        # Node embeddings
        h1 = self.conv1(x, edge_index, edge_attr)
        # h1.shape = [batch_size, dim_h=64]
        h2 = self.conv2(h1, edge_index, edge_attr)
        h3 = self.conv3(h2, edge_index, edge_attr)
        h4 = self.conv4(h3, edge_index, edge_attr)

        # Graph-level readout
        h1 = global_add_pool(h1, batch_index)
        h2 = global_add_pool(h2, batch_index)
        h3 = global_add_pool(h3, batch_index)
        h4 = global_add_pool(h4, batch_index)

        # Concatenate graph embeddings
        # skip_features = graph_level_feats.reshape(h1.shape[0], -1).type(torch.float32)
        h = torch.cat((h1, h2, h3, h4), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.25, training=self.training)
        h = self.lin2(h)

        # output_pos = self.lin2(h)
        # output_al = self.lin2(h)

        return torch.nn.Softplus()(h)
        # return [output_pos, output_al]
