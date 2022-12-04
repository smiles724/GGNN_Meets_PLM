import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("..")
from molformer.tr_spe import Embeddings, FeedForward
from molformer.tr_spe import LayerNorm, SublayerConnection, clones, Generator3D

_NUM_RESIDUE_TYPES = 22
_MAX_SEQUENCE_LENGTH = 9000


def TOYModel(label='dist', N=2, embed_dim=32, head=4, dropout=0.1):
    c = copy.deepcopy
    tgt = 1 if label == 'dist' else _MAX_SEQUENCE_LENGTH
    model = MultiRepresentationTransformer3D(Encoder(EncoderLayer(embed_dim, c(MultiHeadedAttention(head, embed_dim)), c(FeedForward(embed_dim, embed_dim, dropout)), dropout), N),
                                             Embeddings(embed_dim, _NUM_RESIDUE_TYPES), Generator3D(embed_dim, tgt, dropout))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def PSRModel(plm=False, N=2, embed_dim=32, head=4, dropout=0.1):
    c = copy.deepcopy
    if plm:
        embed = None
        embed_dim = 1280
    else:
        embed = Embeddings(embed_dim, _NUM_RESIDUE_TYPES)
    model = MultiRepresentationTransformer3D(Encoder(EncoderLayer(embed_dim, c(MultiHeadedAttention(head, embed_dim)), c(FeedForward(embed_dim, embed_dim, dropout)), dropout), N),
                                             Generator3D(embed_dim, tgt=1, dropout=dropout, mean=True), embed)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def LBAModel(plm=False, N=2, embed_dim=32, head=4, dropout=0.1):
    c = copy.deepcopy
    if plm:
        embed_dim = 1280
        weights_freeze = torch.zeros(20, embed_dim)                             # 20 residue types
        weights_train = torch.nn.parameter.Parameter(torch.rand(2, embed_dim))  # 1 type for ligand and 1 for padding token
        embed = torch.cat((weights_freeze, weights_train), 0).cuda()            # freeze the protein residue embedding
    else:
        embed = Embeddings(embed_dim, _NUM_RESIDUE_TYPES)
    model = MultiRepresentationTransformer3D(Encoder(EncoderLayer(embed_dim, c(MultiHeadedAttention(head, embed_dim)), c(FeedForward(embed_dim, embed_dim, dropout)), dropout), N),
                                             Generator3D(embed_dim, tgt=1, dropout=dropout, mean=True), embed)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def PPIModel(plm=False, N=2, embed_dim=32, head=4, dropout=0.1):
    c = copy.deepcopy
    if plm:
        embed_dim = 1280
        weights_freeze = torch.zeros(20, embed_dim)                             # 20 residue types
        weights_train = torch.nn.parameter.Parameter(torch.rand(2, embed_dim))  # 1 type for ligand and 1 for padding token
        embed = torch.cat((weights_freeze, weights_train), 0).cuda()            # freeze the protein residue embedding
    else:
        embed = Embeddings(embed_dim, _NUM_RESIDUE_TYPES)
    model = MultiRepresentationTransformer3D(Encoder(EncoderLayer(embed_dim, c(MultiHeadedAttention(head, embed_dim)), c(FeedForward(embed_dim, embed_dim, dropout)), dropout), N),
                                             Generator3D(embed_dim, tgt=1, dropout=dropout, mean=False, binary=True), embed)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class MultiRepresentationTransformer3D(nn.Module):
    def __init__(self, encoder, generator, src_embed):
        super(MultiRepresentationTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, batch, mask_id=21):
        src, coords = batch.nodes, batch.coords
        dist = torch.cdist(coords, coords)
        src_mask = (src != mask_id)
        if self.src_embed is None:
            inp = batch.token_reps
        elif type(self.src_embed) == Embeddings:
            inp = self.src_embed(src)
        else:
            inp = F.embedding(src, self.src_embed)
            inp[:, :batch.token_reps.shape[1]] = batch.token_reps
        return self.generator(self.encoder(inp, dist, src_mask.unsqueeze(1)), src_mask)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, dist, mask):
        for layer in self.layers:
            x = layer(x, dist, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """MultiRelationEncoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, dist, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist, mask))
        return self.sublayer[1](x, self.feed_forward)


#######################################
## attention part
#######################################

def attention(query, key, value, dist_conv, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores *= dist_conv

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        # four linear layers
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)

        # 1 * 1 convolution operator
        self.cnn = nn.Sequential(nn.Conv2d(1, h, kernel_size=1), nn.ReLU(), nn.Conv2d(h, h, kernel_size=1))

        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embed_dim // h
        self.h = h
        self.attn = None

    def forward(self, query, key, value, dist_conv, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from (B, N, d) => (B, head, N, d // h)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        dist_conv = self.cnn(dist_conv.unsqueeze(1))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dist_conv, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
