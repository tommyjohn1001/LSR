import torch_geometric.nn as pyg_nn
from all_packages import *
from torch_sparse import SparseTensor


class GraphConvLayer(nn.Module):
    """A GCN module operated on dependency graphs."""

    def __init__(self, mem_dim, layers, dropout, self_loop=False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = dropout

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.linear_output = self.linear_output.cuda()
        self.self_loop = self_loop

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out


class MultiGraphConvLayer(nn.Module):
    """A GCN module operated on multihead attention"""

    def __init__(self, mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = dropout

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[:, i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out


class CustomResGatedGCN(pyg_nn.conv.ResGatedGraphConv):
    def __init__(self, hidden_size, n_layers, dropout_gcn):
        super().__init__(hidden_size, hidden_size)

        self.lin_layers = nn.ModuleList()
        for _ in range(n_layers):

            self.lin_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.LayerNorm(hidden_size),
                )
            )

    def forward(self, adj, gcn_inputs):
        # N: no. nodes
        # adj: [bz, N, N]
        # gcn_inputs: [bz, N, hid_dim]

        _, N, hid_dim = gcn_inputs.size()

        #####################################################################
        ## Batchify adjacency matrices and node embeddings
        #####################################################################
        node_feats = gcn_inputs.reshape(-1, hid_dim)
        # [bz*N, hid_dim]

        # Apply direct sum to adjacency matrix of each mini-batch to create
        # big block diagonal adjacency matrix
        big_adj = torch.block_diag(*list(adj))
        # [bz*N, bz*N]

        # Convert to spare tensor
        big_adj = SparseTensor.from_dense(big_adj)

        #####################################################################
        ## Apply ResGatedGCN
        #####################################################################
        out = node_feats
        for i, lin_layer in enumerate(self.lin_layers):
            out = super().forward(out, big_adj)
            # [bz*N, hid_dim]

            # print(f"out: min {out.min()} - max {out.max()}")

            if NaNReporter.check_abnormal(out, "out"):
                print(f"inside ResGate at {i}")
                exit()

            out = lin_layer(out)
            # [bz*N, hid_dim]

        out = out.reshape(-1, N, hid_dim)
        # [bz, N, hid_dim]

        return out
