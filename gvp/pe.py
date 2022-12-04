import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.utils import to_dense_adj, degree
from scipy import sparse as sp


def Laplacian_position_encoding(edge_index, num_nodes, pos_enc_dim):
    """
    from https://github.com/graphdeeplearning/benchmarking-gnns/blob/b6c407712fa576e9699555e1e035d1e327ccae6c/data/molecules.py
    modified from DGL to torch_geometric
    """
    with torch.no_grad():
        adj = to_dense_adj(edge_index).squeeze(0)

        N = sp.diags(degree(edge_index[0]).numpy().clip(1) ** -0.5, dtype=float)
        L = sp.eye(num_nodes) - N * adj * N    # tensor可以直接与array相乘，不必用csr_matrix
        EigVal, EigVec = np.linalg.eig(np.asarray(L))
        idx = EigVal.argsort()     # increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
        pe = torch.from_numpy(EigVec[:, 1: pos_enc_dim + 1]).float()
    return pe


#############################################################################################

def random_walk_position_encoding(edge_index, pos_enc_dim):
    """
    from https://github.com/vijaydwivedi75/gnn-lspe/blob/21d2a654569fcac42cde28ce0695386d1c56a9b9/data/molecules.py
    """
    with torch.no_grad():
        adj = to_dense_adj(edge_index).squeeze(0).numpy()
        Dinv = sp.diags(degree(edge_index[0]).numpy().clip(1) ** -1.0, dtype=float)  # D^-1
        M = adj @ Dinv    # AD^-1

        PE = [torch.from_numpy(M.diagonal().copy()).float()]
        M_power = M
        for _ in range(pos_enc_dim - 1):
            M_power = M_power @ M    # 注意不是 * 而是 @
            PE.append(torch.from_numpy(M_power.diagonal().copy()).float())
        PE = torch.stack(PE, dim=-1)  # (N,D)
    return PE


##############################################################################################


class PositionalEncoding1D(nn.Module):
    def __init__(self, dim):
        """
        from https://github.com/tatp22/multidim-positional-encoding
        dim: The last dimension of the pe_id you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.orig_dim = dim
        self.dim = dim
        if self.dim % 2 != 0: self.dim += 1  # 奇数+1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))  # (1,d/2)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pe_id):
        pos_x = pe_id.type(self.inv_freq.type())   # (N,1)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)  # 外积，(N,d/2)
        emb_x = torch.stack((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)   # (N,d/2,2)
        emb_x = torch.flatten(emb_x, -2, -1)    # (N,d)
        return emb_x[:, :self.orig_dim]


class PositionalEncoding(nn.Module):
    """
    Harvard's reproduction. This sinusoidal PE is designed for fixed-length sequence.
    """
    def __init__(self, embed_dim, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 调整PE的维度，并将其存放在不视为模型参数的缓冲区内
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 让token embedding与PE直接相加
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

