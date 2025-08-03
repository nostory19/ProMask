from torch_geometric.nn.norm import GraphNorm, GraphSizeNorm
from torch_geometric.nn.glob.glob import global_mean_pool, global_add_pool, global_max_pool
from .utils import pad2batch
import torch_scatter as scatter
import torch
import torch.nn as nn
from torch_geometric.nn import GraphNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse


class Seq(nn.Module):
    '''
    An extension of nn.Sequential.
    '''

    def __init__(self, modlist):
        super().__init__()
        self.modlist = nn.ModuleList(modlist)

    def forward(self, *args, **kwargs):
        out = self.modlist[0](*args, **kwargs)
        for i in range(1, len(self.modlist)):
            out = self.modlist[i](out)
        return out


class MLP(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 dropout=0,
                 tail_activation=False,
                 activation=nn.ReLU(inplace=True),
                 gn=False):
        super().__init__()
        modlist = []
        self.seq = None
        if num_layers == 1:
            modlist.append(nn.Linear(input_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)
        else:
            modlist.append(nn.Linear(input_channels, hidden_channels))
            for _ in range(num_layers - 2):
                if gn:
                    modlist.append(GraphNorm(hidden_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
                modlist.append(nn.Linear(hidden_channels, hidden_channels))
            if gn:
                modlist.append(GraphNorm(hidden_channels))
            if dropout > 0:
                modlist.append(nn.Dropout(p=dropout, inplace=True))
            modlist.append(activation)
            modlist.append(nn.Linear(hidden_channels, output_channels))
            if tail_activation:
                if gn:
                    modlist.append(GraphNorm(output_channels))
                if dropout > 0:
                    modlist.append(nn.Dropout(p=dropout, inplace=True))
                modlist.append(activation)
            self.seq = Seq(modlist)

    def forward(self, x):
        return self.seq(x)


def buildAdj(edge_index, edge_weight, n_node: int, aggr: str):
    '''
        Calculating the normalized adjacency matrix.
        Args:
            n_node: number of nodes in graph.
            aggr: the aggregation method, can be "mean", "sum" or "gcn".
        '''
    adj = torch.sparse_coo_tensor(edge_index,
                                  edge_weight,
                                  size=(n_node, n_node))
    deg = torch.sparse.sum(adj, dim=(1,)).to_dense().flatten()
    deg[deg < 0.5] += 1.0
    if aggr == "mean":
        deg = 1.0 / deg
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "sum":
        return torch.sparse_coo_tensor(edge_index,
                                       edge_weight,
                                       size=(n_node, n_node))
    elif aggr == "gcn":
        deg = torch.pow(deg, -0.5)
        return torch.sparse_coo_tensor(edge_index,
                                       deg[edge_index[0]] * edge_weight *
                                       deg[edge_index[1]],
                                       size=(n_node, n_node))
    else:
        raise NotImplementedError


class DynamicProtoMask(nn.Module):
    '''
        Module for generating prototype masks
    '''

    def __init__(self,
                 in_channels,
                 max_deg,
                 aggr="mean"
                 ):
        super().__init__()
        self.lin = nn.Linear(in_channels, in_channels, bias=False)
        self.aggr = aggr
        self.input_emb = nn.Embedding(
            max_deg + 1,
            in_channels,
            scale_grad_by_freq=False
        )
        self.emb_gn = GraphNorm(in_channels)
        self.att_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)  # 输出一个标量：单个注意力分数
        )

        self.score_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        self.temp = nn.Parameter(torch.tensor(10.0))
        self.adj = torch.sparse_coo_tensor(size=(0, 0))

    def forward(self, x, edge_index, edge_weight, subG_node, tau=0.7, top_k=None):
        x_proj = self.lin(x)  # (21521, 64)

        if isinstance(subG_node, torch.Tensor):
            valid = subG_node >= 0  # Boolean(80, 160)
            batch_idx = torch.repeat_interleave(
                torch.arange(subG_node.size(0), device=x.device),
                valid.sum(-1))  # (728)
            node_idx = subG_node[valid]
            # subgraph_iter = enumerate(subG_node)

        else:
            batch_idx = torch.cat([torch.full_like(g, i)
                                   for i, g in enumerate(subG_node)])
            node_idx = torch.cat(subG_node)
        n_node = x.shape[0]
        att_score = self.att_mlp(x_proj[node_idx]).squeeze(-1)
        att_score = scatter.scatter_softmax(att_score, batch_idx)
        # Calculate the weighted prototype
        weighted_x = x_proj[node_idx] * att_score.unsqueeze(-1)
        proto = scatter.scatter(weighted_x,
                                batch_idx,
                                dim=0,
                                dim_size=batch_idx.max() + 1,
                                reduce="sum")
        x_norm = F.normalize(x_proj, dim=-1)  # (21521, 64)
        proto_norm = F.normalize(proto, dim=-1)  # (80, 64)
        sim_mat = x_norm @ proto_norm.t()

        hard_mask = torch.zeros(x.size(0), device=x.device)
        hard_mask[node_idx] = 1.0

        if top_k is None:
            percent = 0.99  # 0.9 ->top 10%
            thresh = torch.quantile(sim_mat.max(dim=1).values, percent)
            candidate = (sim_mat.max(dim=1).values > thresh).float()
            candidate_numpy = candidate.detach().cpu().numpy()

            sim_max_score, sim_argmax = sim_mat.max(dim=1)
            candidate_idx = (candidate == 1.0).nonzero(as_tuple=False).squeeze(-1)
            candidate_subgraph_assignment = sim_argmax[candidate_idx]  # [num_candidate_nodes]
        else:
            candidate = torch.zeros_like(hard_mask)
            for b in range(proto.size(0)):
                ext_mask = (hard_mask == 0).bool()
                ext_scores = sim_mat[:, b] * ext_mask
                top_idx = torch.topk(ext_scores, k=top_k).indices

        final_mask = hard_mask + (1 - hard_mask) * candidate  # (N,)

        return final_mask.unsqueeze(-1), proto, candidate_idx, candidate_subgraph_assignment  # (N,1)


class MaskConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for ProMaskNet.
    We use different parameters to transform the features of node with different labels individually, and mix them.
    Args:
        aggr: the aggregation method.
        z_ratio: the ratio to mix the transformed features.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean",
                 z_ratio=0.8,
                 dropout=0.2):
        super().__init__()
        self.trans_fns = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.comb_fns = nn.ModuleList([
            nn.Linear(in_channels + out_channels, out_channels),
            nn.Linear(in_channels + out_channels, out_channels)
        ])
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)
        self.z_ratio = z_ratio
        self.reset_parameters()
        self.dropout = dropout

    def reset_parameters(self):
        for _ in self.trans_fns:
            _.reset_parameters()
        for _ in self.comb_fns:
            _.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight, mask):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
            # transform node features with different parameters individually.
        x1 = self.activation(self.trans_fns[1](x_))
        x0 = self.activation(self.trans_fns[0](x_))
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        # pass messages.
        x = self.adj @ x
        x = self.gn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.cat((x, x_), dim=-1)
        # transform node features with different parameters individually.
        x1 = self.comb_fns[1](x)
        x0 = self.comb_fns[0](x)
        # mix transformed feature.
        x = torch.where(mask, self.z_ratio * x1 + (1 - self.z_ratio) * x0,
                        self.z_ratio * x0 + (1 - self.z_ratio) * x1)
        return x


class PPRDiffusedGCN(nn.Module):
    def __init__(self, channels, num_layers, alpha=0.15, k=10, use_residual=True):
        super().__init__()
        self.gcn = GCNConv(channels, channels * num_layers)
        self.layer_norm = nn.LayerNorm(channels * num_layers)
        self.graph_norm = GraphNorm(channels * num_layers)
        self.act = nn.ELU()
        self.alpha = alpha
        self.k = k
        self.use_residual = use_residual

    def forward(self, x, edge_index, batch=None):
        """
        :param x: Node features [N, d]
        :param edge_index: [2, E]
        :param batch: Node-to-graph assignment [N], optional (required for GraphNorm)
        """
        num_nodes = x.size(0)
        device = x.device

        # === 1. Build dense adj
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # [N, N]
        adj = adj + torch.eye(num_nodes, device=device)

        deg = adj.sum(dim=-1, keepdim=True)
        deg[deg == 0] = 1.0
        adj_norm = adj / deg

        # === 2. PPR approximation
        ppr = torch.zeros_like(adj_norm)
        mat = torch.eye(num_nodes, device=device)
        for i in range(self.k):
            mat = torch.matmul(adj_norm, mat)
            ppr += (1 - self.alpha) * (self.alpha ** i) * mat
        ppr += self.alpha * torch.eye(num_nodes, device=device)

        # === 3. Sparsify
        ppr[ppr < 1e-4] = 0.0
        edge_index_ppr, edge_weight = dense_to_sparse(ppr)

        # === 4. GCN
        out = self.gcn(x, edge_index_ppr, edge_weight)

        # === 5. Normalization + Activation
        out = self.layer_norm(out)
        # batch = batch.to(out.device)
        # if batch is not None:
        # out = self.graph_norm(out, batch)
        out = self.act(out)

        return out


class EmbMaskConv(nn.Module):
    '''
    combination of some MaskConv layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''

    def __init__(self,
                 hidden_channels,
                 output_channels,
                 num_layers,
                 max_deg,
                 dropout=0,
                 activation=nn.ReLU(),
                 conv=MaskConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1,
                                      hidden_channels,
                                      scale_grad_by_freq=False)
        self.emb_gn = GraphNorm(hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        for _ in range(num_layers - 1):
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=hidden_channels,
                     activation=activation,
                     **kwargs))
        self.convs.append(
            conv(in_channels=hidden_channels,
                 out_channels=output_channels,
                 activation=activation,
                 **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
            if self.jk:
                self.gns.append(
                    GraphNorm(output_channels +
                              (num_layers - 1) * hidden_channels))
            else:
                self.gns.append(GraphNorm(output_channels))
        else:
            self.gns = None

        self.sub_mask = DynamicProtoMask(in_channels=hidden_channels,
                                         max_deg=max_deg)  # (in_channels:64, out: 64)
        self.virtual_gnn = GCNConv(hidden_channels, hidden_channels)
        self.reset_parameters()
        self.ppr_diffused_gnn = PPRDiffusedGCN(hidden_channels, num_layers, alpha=0.2, k=8)

    def reset_parameters(self):
        self.input_emb.reset_parameters()
        self.emb_gn.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, subG_node, edge_index_virtual, z=None):
        # convert integer input to vector node features.
        x = self.input_emb(x).reshape(x.shape[0], -1)  # (N,64)
        x = self.emb_gn(x)  # (N, 64)
        mask_soft, proto, candidate_idx, candidate_subgraph_assignment = self.sub_mask(x, edge_index, edge_weight,
                                                                                       subG_node)
        mask_soft = mask_soft.reshape(-1, 1)
        proto_updated = self.ppr_diffused_gnn(proto, edge_index_virtual, batch=torch.arange(proto.size(0)))
        mask = (mask_soft > 0.5)
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        # pass messages at each layer. 
        for layer, conv in enumerate(self.convs[:-1]):
            proto_updated = self.ppr_diffused_gnn(proto, edge_index_virtual, batch=torch.arange(proto.size(0)))
            x = conv(x, edge_index, edge_weight, mask)
            mask_soft, proto, candidate_idx, candidate_subgraph_assignment = self.sub_mask(x, edge_index, edge_weight,
                                                                                           subG_node)
            mask_soft = mask_soft.reshape(-1, 1)
            mask = (mask_soft > 0.5)
            xs.append(x)
            if not (self.gns is None):
                x = self.gns[layer](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight, mask)
        xs.append(x)

        if self.jk:
            x = torch.cat(xs, dim=-1)
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x, proto_updated, candidate_idx, candidate_subgraph_assignment
        else:
            x = xs[-1]
            if not (self.gns is None):
                x = self.gns[-1](x)
            return x, proto_updated, candidate_idx, candidate_subgraph_assignment


class PoolModule(nn.Module):
    '''
    Modules used for pooling node embeddings to produce subgraph embeddings.
    Args:
        trans_fn: module to transfer node embeddings.
        pool_fn: module to pool node embeddings like global_add_pool.
    '''

    def __init__(self, pool_fn, trans_fn=None):
        super().__init__()
        self.pool_fn = pool_fn
        self.trans_fn = trans_fn

    def forward(self, x, batch):
        # The j-th element in batch vector is i if node j is in the i-th subgraph.
        # for example [0,1,0,0,1,1,2,2] means nodes 0,2,3 in subgraph 0, nodes 1,4,5 in subgraph 1, and nodes 6,7 in subgraph 2.
        if self.trans_fn is not None:
            x = self.trans_fn(x)
        return self.pool_fn(x, batch)


class AddPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_add_pool, trans_fn)


class MaxPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_max_pool, trans_fn)


class MeanPool(PoolModule):
    def __init__(self, trans_fn=None):
        super().__init__(global_mean_pool, trans_fn)


class SizePool(AddPool):
    def __init__(self, trans_fn=None):
        super().__init__(trans_fn)

    def forward(self, x, batch):
        if x is not None:
            if self.trans_fn is not None:
                x = self.trans_fn(x)
        x = GraphSizeNorm()(x, batch)
        return self.pool_fn(x, batch)


class FlexibleMLP(nn.Module):
    def __init__(self, channel_list, act=nn.ELU(), dropout=0.5):
        super().__init__()
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(nn.Linear(channel_list[i], channel_list[i + 1]))
            layers.append(act)
            if isinstance(dropout, list):
                layers.append(nn.Dropout(dropout[i]))
            else:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProMaskNet(nn.Module):
    '''
    ProMaskNet model: combine message passing layers and mlps and pooling layers.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv: EmbMaskConv, preds: nn.ModuleList,
                 pools: nn.ModuleList, hidden_dim, conv_layer):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools
        self.expand_proto = nn.Linear(hidden_dim, hidden_dim * conv_layer)
        self.gate_linear = nn.Linear((hidden_dim * conv_layer) * 2, hidden_dim * conv_layer)
        self.mlp = FlexibleMLP(
            channel_list=[(hidden_dim * conv_layer) * 2, hidden_dim * conv_layer, hidden_dim * conv_layer],
            dropout=[0.5, 0.5])

    def NodeEmb(self, x, edge_index, edge_weight, subG_node, edge_index_virtual, z=None):
        embs = []
        protos = []

        for _ in range(x.shape[1]):
            emb, proto_updated, candidate_idx, candidate_subgraph_assignment = self.conv(
                x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                edge_index, edge_weight, subG_node, edge_index_virtual, z)
            # temp = emb.reshape(emb.shape[0], 1, emb.shape[-1])
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
            protos.append(proto_updated.reshape(proto_updated.shape[0], 1, proto_updated.shape[-1]))
            # protos.append(proto_updated)
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        sub_emb = torch.cat(protos, dim=1)
        sub_emb = torch.mean(sub_emb, dim=1)
        return emb, sub_emb, candidate_idx, candidate_subgraph_assignment

    def Pool(self, emb, sub_emb, subG_node, edge_index, candidate_idx, candidate_subgraph_assignment, pool):

        device = emb.device
        batch, pos = pad2batch(subG_node)
        emb_sub = emb[pos]

        emb_subg_pooled = pool(emb_sub, batch)  #

        # Construct a ragged list of external nodes for similar node features.
        num_subgraphs = len(subG_node)
        comp_node_list = [[] for _ in range(num_subgraphs)]
        candidate_idx = candidate_idx.tolist()  # list(216)
        candidate_subgraph_assignment = candidate_subgraph_assignment.tolist()

        for nid, sid in zip(candidate_idx, candidate_subgraph_assignment):
            comp_node_list[sid].append(nid)

        emb_dim = emb.shape[1]
        emb_comp_pooled = torch.zeros((num_subgraphs, emb_dim), device=device)
        for sid, node_ids in enumerate(comp_node_list):
            if len(node_ids) == 0:
                continue
            nodes = torch.tensor(node_ids, dtype=torch.long, device=device)
            emb_nodes = emb[nodes]
            emb_comp_pooled[sid] = pool(emb_nodes, torch.zeros(len(nodes), dtype=torch.long, device=device))

        emb_combined = torch.cat([emb_subg_pooled, emb_comp_pooled], dim=-1)
        emb = self.mlp(emb_combined)

        return emb, sub_emb

    def build_virtual_graph(self, subG_node):
        if isinstance(subG_node, torch.Tensor):
            # [B, L] - >List[Tensor]
            subG_node = [subG_node[i][subG_node[i] != -1] for i in range(subG_node.size(0))]

        B = len(subG_node)
        edges = []
        for i in range(B):
            for j in range(i + 1, B):
                set_i = set(subG_node[i].tolist())
                set_j = set(subG_node[j].tolist())
                if len(set_i & set_j) > 0:
                    edges.append([i, j])
                    edges.append([j, i])
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        edge_index_virtual = torch.tensor(edges, dtype=torch.long).T
        return edge_index_virtual

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        edge_index_virtual = self.build_virtual_graph(subG_node).to(x.device)
        emb, sub_emb, candidate_idx, candidate_subgraph_assignment = self.NodeEmb(x, edge_index, edge_weight, subG_node,
                                                                                  edge_index_virtual, z)
        emb, sub_emb = self.Pool(emb, sub_emb, subG_node, edge_index, candidate_idx, candidate_subgraph_assignment,
                                 self.pools[id])  # (80, 128)
        return self.preds[id](emb), self.preds[id](sub_emb)  # (, classes)


class MyGCNConv(torch.nn.Module):
    '''
    A kind of message passing layer we use for pretrained GNNs.
    Args:
        aggr: the aggregation method.
    '''

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation=nn.ReLU(inplace=True),
                 aggr="mean"):
        super().__init__()
        self.trans_fn = nn.Linear(in_channels, out_channels)
        self.comb_fn = nn.Linear(in_channels + out_channels, out_channels)
        self.adj = torch.sparse_coo_tensor(size=(0, 0))
        self.activation = activation
        self.aggr = aggr
        self.gn = GraphNorm(out_channels)

    def reset_parameters(self):
        self.trans_fn.reset_parameters()
        self.comb_fn.reset_parameters()
        self.gn.reset_parameters()

    def forward(self, x_, edge_index, edge_weight):
        if self.adj.shape[0] == 0:
            n_node = x_.shape[0]
            self.adj = buildAdj(edge_index, edge_weight, n_node, self.aggr)
        x = self.trans_fn(x_)
        x = self.activation(x)
        x = self.adj @ x
        x = self.gn(x)
        x = torch.cat((x, x_), dim=-1)
        x = self.comb_fn(x)
        return x


class EmbGConv(torch.nn.Module):
    '''
    combination of some message passing layers, normalization layers, dropout layers, and activation function.
    Args:
        max_deg: the max integer in input node features.
        conv: the message passing layer we use.
        gn: whether to use GraphNorm.
        jk: whether to use Jumping Knowledge Network.
    '''

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 num_layers: int,
                 max_deg: int,
                 dropout=0,
                 activation=nn.ReLU(inplace=True),
                 conv=GCNConv,
                 gn=True,
                 jk=False,
                 **kwargs):
        super().__init__()
        self.input_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.convs = nn.ModuleList()
        self.jk = jk
        if num_layers > 1:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=hidden_channels,
                     **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(
                    conv(in_channels=hidden_channels,
                         out_channels=hidden_channels,
                         **kwargs))
            self.convs.append(
                conv(in_channels=hidden_channels,
                     out_channels=output_channels,
                     **kwargs))
        else:
            self.convs.append(
                conv(in_channels=input_channels,
                     out_channels=output_channels,
                     **kwargs))
        self.activation = activation
        self.dropout = dropout
        if gn:
            self.gns = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.gns.append(GraphNorm(hidden_channels))
        else:
            self.gns = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not (self.gns is None):
            for gn in self.gns:
                gn.reset_parameters()

    def forward(self, x, edge_index, edge_weight, z=None):
        xs = []
        x = F.dropout(self.input_emb(x.reshape(-1)),
                      p=self.dropout,
                      training=self.training)
        for layer, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            if not (self.gns is None):
                x = self.gns[layer](x)
            xs.append(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(self.convs[-1](x, edge_index, edge_weight))
        if self.jk:
            return torch.cat(xs, dim=-1)
        else:
            return xs[-1]


class EdgeGNN(nn.Module):
    '''
    EdgeGNN model: combine message passing layers and mlps and pooling layers to do link prediction task.
    Args:
        preds and pools are ModuleList containing the same number of MLPs and Pooling layers.
        preds[id] and pools[id] is used to predict the id-th target. Can be used for SSL.
    '''

    def __init__(self, conv, preds: nn.ModuleList, pools: nn.ModuleList):
        super().__init__()
        self.conv = conv
        self.preds = preds
        self.pools = pools

    def NodeEmb(self, x, edge_index, edge_weight, z=None):
        embs = []
        for _ in range(x.shape[1]):
            emb = self.conv(x[:, _, :].reshape(x.shape[0], x.shape[-1]),
                            edge_index, edge_weight, z)
            embs.append(emb.reshape(emb.shape[0], 1, emb.shape[-1]))
        emb = torch.cat(embs, dim=1)
        emb = torch.mean(emb, dim=1)
        return emb

    def Pool(self, emb, subG_node, pool):
        emb = emb[subG_node]
        emb = torch.mean(emb, dim=1)
        return emb

    def forward(self, x, edge_index, edge_weight, subG_node, z=None, id=0):
        emb = self.NodeEmb(x, edge_index, edge_weight, z)
        emb = self.Pool(emb, subG_node, self.pools[id])
        return self.preds[id](emb)
