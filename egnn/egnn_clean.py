import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter

from gvp.atom3d import _NUM_RESIDUE_TYPES


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf), act_fn, nn.Linear(hidden_nf, hidden_nf), act_fn)
        self.node_mlp = nn.Sequential(nn.Linear(hidden_nf + input_nf, hidden_nf), act_fn, nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:               # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index[0], edge_index[1]
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr


class EGNN(nn.Module):
    def __init__(self, vocab=_NUM_RESIDUE_TYPES, hidden_nf=32, in_edge_nf=16, device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False,
                 normalize=False, tanh=False, plm=False):                           # edge feature dimension is num_rbf=16
        """
        :param vocab: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        """

        super(EGNN, self).__init__()
        self.plm = plm
        if self.plm:
            self.hidden_nf = 1280
        else:
            self.hidden_nf = hidden_nf
            self.embed = nn.Embedding(vocab, self.hidden_nf)

        self.device = device
        self.n_layers = n_layers
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i,
                            E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, attention=attention,
                                  normalize=normalize, tanh=tanh))
        self.to(self.device)

    def forward(self, batch):
        if self.plm:
            x, h = batch.x, batch.plm
        else:
            x, h = batch.x, self.embed(batch.atoms)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, batch.edge_index, x, edge_attr=batch.edge_s)
        return h


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    """ fully-connected graphs """
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


class TOYModel(EGNN):

    def __init__(self, label='dist', **kwargs):
        super().__init__(**kwargs)
        if label == 'id':
            _MAX_SEQUENCE_LENGTH = 9000
            self.dense = nn.Sequential(nn.Linear(self.hidden_nf, 2 * self.hidden_nf), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * self.hidden_nf, _MAX_SEQUENCE_LENGTH))
        else:
            self.dense = nn.Sequential(nn.Linear(self.hidden_nf, 2 * self.hidden_nf), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * self.hidden_nf, 1), nn.ReLU(), nn.Flatten())

    def forward(self, batch, level='residue'):
        out = self.dense(super().forward(batch))
        if level == 'atom':
            idx = [batch.ca_idx[0]]
            for i in range(1, len(batch.ca_idx)):
                idx.append(batch.ca_idx[i] + batch.ptr[i].item())
            idx = np.concatenate(idx)
            return out[idx]
        else:
            return out.squeeze(-1)


class BaseModel(EGNN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Sequential(nn.Linear(self.hidden_nf, 2 * self.hidden_nf), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * self.hidden_nf, 1))

    def forward(self, batch):
        out = super().forward(batch)
        batch_id = batch.batch

        out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        out = self.dense(out).squeeze(-1)
        return out


PSRModel = BaseModel


class LBAModel(EGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Sequential(nn.Linear(self.hidden_nf, 2 * self.hidden_nf), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * self.hidden_nf, 1))
        if self.plm:
            weights_freeze = torch.zeros(20, self.hidden_nf)                                  # 20 residue types
            weights_train = torch.nn.parameter.Parameter(torch.rand(1, self.hidden_nf))       # 1 type for ligand
            self.embed = torch.cat((weights_freeze, weights_train), 0).cuda()

    def forward(self, batch):
        if self.plm:
            x, h = batch.x, batch.plm + F.embedding(batch.atoms, self.embed)
        else:
            x, h = batch.x, self.embed(batch.atoms)

        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, batch.edge_index, x, edge_attr=batch.edge_s)
        out = torch_scatter.scatter_mean(h, batch.batch, dim=0)
        self.dense(out).squeeze(-1)


class PPIModel(EGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = nn.Sequential(nn.Linear(self.hidden_nf, 2 * self.hidden_nf), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * self.hidden_nf, 1))

    def forward(self, batch):
        graph1, graph2 = batch
        out1 = self.dense(super().forward(graph1))
        out2 = self.dense(super().forward(graph2))
        return torch.sigmoid(torch.cat([out1, out2]).squeeze(-1))
