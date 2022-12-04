import math
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
import torch_geometric
import torch_scatter
from atom3d.datasets import LMDBDataset
from torch.utils.data import IterableDataset, Dataset
import esm

from . import GVP, GVPConvLayer, LayerNorm
from .data import _normalize, _rbf

_NUM_ATOM_TYPES = 9
_NUM_RESIDUE_TYPES = 22
_MAX_SEQUENCE_LENGTH = 9000
_element_mapping = lambda x: {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'S': 5, 'Cl': 6, 'CL': 6, 'P': 7}.get(x, 8)
_amino_acids = lambda x: {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                          'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'LIG': 20}.get(x, 21)  # 'LIG' for small-molecule ligand
RESTYPE_3to1 = lambda x: {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                          'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}.get(x, '<unk>')    # strange residues (e.g., CA, BET)
_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))
    return edge_s, edge_v


def get_plm_reps(df, model, converter):
    seq = ''.join([RESTYPE_3to1(i) for i in list(df['resname'])])
    batch_tokens = converter([("_", seq)])[2]  # [label, sequence]
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33], return_contacts=True)  # Extract per-residue representations
    token_reps = results["representations"][33][:, 1:-1][0].detach()                  # the head and tail tokens are placeholder
    return token_reps


class BaseTransform:
    """
    Implementation of an ATOM3D Transform which featurizes the atomic coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.

    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]

    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    """

    def __init__(self, edge_cutoff=8, num_rbf=16, connection='rball', device='cpu'):  # cut_off = 4.5 for atom-level, 8 for residue-level graphs
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.connection = connection

    def __call__(self, df, level='residue'):
        """
        :param df: `pandas.DataFrame` of atomic coordinates in the ATOM3D format
        :return: `torch_geometric.data.Data` structure graph
        """
        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
            if level == 'residue':
                nodes = torch.as_tensor(list(map(_amino_acids, df.resname)), dtype=torch.long, device=self.device)
            else:
                nodes = torch.as_tensor(list(map(_element_mapping, df.element)), dtype=torch.long, device=self.device)

            # some proteins are added by HIS or miss some residues
            if self.connection == 'knn':
                edge_index = torch_cluster.knn_graph(coords, k=10)
            else:
                edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)  # r-ball graph
            edge_s, edge_v = _edge_features(coords, edge_index, D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)  # use RBF to represent distance
            return torch_geometric.data.Data(x=coords, atoms=nodes, edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)


class BaseModel(nn.Module):
    """
    A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch` and returns a single scalar.
    """

    def __init__(self, num_rbf=16, plm=False):
        super().__init__()
        activations = (F.relu, None)
        _EMBED_DIM = 32
        self.plm = plm
        if self.plm:
            _EMBED_DIM = 1280
        else:
            self.embed = nn.Embedding(_NUM_RESIDUE_TYPES, _EMBED_DIM)  # the dimension of node embedding
        self.dropout = nn.Dropout(p=0.7)

        self.W_e = nn.Sequential(LayerNorm((num_rbf, 1)), GVP((num_rbf, 1), _DEFAULT_E_DIM, activations=(None, None), vector_gate=True))
        self.W_v = nn.Sequential(LayerNorm((_EMBED_DIM, 0)), GVP((_EMBED_DIM, 0), _DEFAULT_V_DIM, activations=(None, None), vector_gate=True))

        self.layers = nn.ModuleList(GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, activations=activations, vector_gate=True) for _ in range(5))

        ns, _ = _DEFAULT_V_DIM
        self.W_out = nn.Sequential(LayerNorm(_DEFAULT_V_DIM), GVP(_DEFAULT_V_DIM, (ns, 0), activations=activations, vector_gate=True))
        self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, 1))  # output scalar

    def forward(self, batch, scatter_mean=True, dense=True):
        """
        :param batch: `torch_geometric.data.Batch` with data attributes as returned from a BaseTransform
        :param scatter_mean: if `True`, returns mean of final node embeddings (for each graph), else, returns embeddings seperately
        :param dense: if `True`, applies final dense layer to reduce embedding to a single scalar; else, returns the embedding
        """
        if self.plm:
            h_V = batch.plm
        else:
            h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        batch_id = batch.batch
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if scatter_mean: out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense: out = self.dense(out).squeeze(-1)
        return out


class PSRTransform(BaseTransform):
    """
    Transforms dict-style entries from the ATOM3D PSR dataset to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the GDT_TS, `id` for the name of the target, and all structural attributes as described in BaseTransform.
    Residue-level graphs.
    """

    def __init__(self, plm=False, **kwargs):
        super().__init__(**kwargs)
        self.plm = plm
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load the latest ESM-2
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __call__(self, elem, index_filter=False):
        df = elem['atoms']
        if index_filter: df = df[df.residue > 0]  # ensure residue index > 0
        df_ca = df[df['name'] == 'CA']            # only CA to build residue-level graph
        data = super().__call__(df_ca)            # use parent's function to process df
        data.label = elem['scores']['gdt_ts']
        data.id = eval(elem['id'])[0]
        if self.plm: data.plm = get_plm_reps(df_ca, self.model, self.batch_converter)
        return data


class PSRDataset(Dataset):

    def __init__(self, lmdb_dataset, plm=False, device='cpu'):
        self.dataset = lmdb_dataset
        self.device = device
        self.plm = plm
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load the latest ESM-2
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        with torch.no_grad():
            df = self.dataset[item]['atoms']
            df = df[df['name'] == 'CA']
            nodes = torch.as_tensor(list(map(_amino_acids, df.resname)), dtype=torch.long, device=self.device)
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
            label = torch.as_tensor(self.dataset[item]['scores']['gdt_ts'], device=self.device)
        id = eval(self.dataset[item]['id'])[0]
        if self.plm:
            token_reps = get_plm_reps(df, self.model, self.batch_converter)
            return nodes, coords, (label, id), token_reps
        return nodes, coords, (label, id)


class PPITransform(BaseTransform):  # IterableDataset cannot achieve this goal with PLM, since the gradient disappear

    def __init__(self, plm=False, cutoff=8, **kwargs):
        super().__init__(**kwargs)
        self.plm = plm
        self.cutoff = cutoff
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load the latest ESM-2
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __call__(self, elem, index_filter=False):
        pairs = elem['atoms_pairs']
        pairs = pairs[pairs['name'] == 'CA']

        subunits = pairs['subunit'].unique()
        bound1, bound2 = pairs[pairs['subunit'] == subunits[0]], pairs[pairs['subunit'] == subunits[1]]
        graph1, graph2 = super().__call__(bound1), super().__call__(bound2)
        dist = torch.cdist(graph1.x, graph2.x) < self.cutoff

        graph1.label = (torch.sum(dist, dim=-1) > 0).float()
        graph2.label = (torch.sum(dist, dim=0) > 0).float()
        if self.plm:
            token_reps1 = get_plm_reps(bound1, self.model, self.batch_converter)
            token_reps2 = get_plm_reps(bound2, self.model, self.batch_converter)
            graph1.plm, graph2.plm = token_reps1, token_reps2
        return graph1, graph2


class PPIDataset(Dataset):

    def __init__(self, lmdb_dataset, plm=False, cutoff=8.0, device='cpu'):
        self.dataset = lmdb_dataset
        self.device = device
        self.cutoff = cutoff
        self.plm = plm
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        with torch.no_grad():
            pairs = self.dataset[item]['atoms_pairs']
            pairs = pairs[pairs['name'] == 'CA']
            subunits = pairs['subunit'].unique()
            bound1, bound2 = pairs[pairs['subunit'] == subunits[0]], pairs[pairs['subunit'] == subunits[1]]

            nodes1 = torch.as_tensor(list(map(_amino_acids, bound1.resname)), dtype=torch.long, device=self.device)
            nodes2 = torch.as_tensor(list(map(_amino_acids, bound2.resname)), dtype=torch.long, device=self.device)

            coords1 = torch.as_tensor(bound1[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
            coords2 = torch.as_tensor(bound2[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)

            dist = torch.cdist(coords1, coords2) < self.cutoff
            label1 = (torch.sum(dist, dim=-1) > 0).float()
            label2 = (torch.sum(dist, dim=0) > 0).float()
            if self.plm:
                token_reps1 = get_plm_reps(bound1, self.model, self.batch_converter)
                token_reps2 = get_plm_reps(bound2, self.model, self.batch_converter)
                return (nodes1, nodes2), (coords1, coords2), (label1, label2), (token_reps1, token_reps2)
        return (nodes1, nodes2), (coords1, coords2), (label1, label2)


class PPIModel(BaseModel):
    """
    Accept a tuple (batch1, batch2) of `torch_geometric.data.Batch` graphs, where each graph index in a batch is paired with the same graph index in the other batch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM

    def forward(self, batch, level='residue', **kwargs):
        out1 = super().forward(batch[0], scatter_mean=False, dense=True)
        out2 = super().forward(batch[1], scatter_mean=False, dense=True)
        out = torch.cat([out1, out2], dim=-1)
        if level == 'atom': out = out[batch.ca_idx + batch.ptr[:-1]]
        return torch.sigmoid(out)


class LBATransform(BaseTransform):
    """
    Transforms dict-style entries from the ATOM3D LBA dataset to featurized graphs. Returns a `torch_geometric.data.Data`
    graph with attribute `label` for the neglog-affinity and all structural attributes as described in BaseTransform.
    
    The transform combines the atomic coordinates of the pocket and ligand atoms and treats them as a single structure / graph.
    Includes hydrogen atoms.
    """

    def __init__(self, plm=False, **kwargs):
        super().__init__(**kwargs)
        self.plm = plm
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load the latest ESM-2
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __call__(self, elem, pocket=True):
        if pocket:
            protein, ligand = elem['atoms_pocket'], elem['atoms_ligand']
        else:                        # whole protein is very long
            protein, ligand = elem['atoms_protein'], elem['atoms_ligand']
        protein_ca = protein[protein['name'] == 'CA']
        df = pd.concat([protein_ca, ligand], ignore_index=True)
        data = super().__call__(df)  # heterogeneous graphs
        with torch.no_grad():
            data.label = elem['scores']['neglog_aff']
            lig_flag = torch.zeros(df.shape[0], device=self.device, dtype=torch.bool)
            lig_flag[-len(ligand):] = 1
            data.lig_flag = lig_flag

        if self.plm:
            protein_reps = get_plm_reps(protein_ca, self.model, self.batch_converter)
            token_reps = torch.zeros((len(df), protein_reps.shape[-1])).cuda()
            token_reps[:len(protein_ca)] = protein_reps
            data.plm = token_reps
        return data


class LBADataset(Dataset):

    def __init__(self, lmdb_dataset, plm=False, device='cpu'):
        self.dataset = lmdb_dataset
        self.device = device
        self.plm = plm
        if self.plm:
            self.model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load the latest ESM-2
            self.batch_converter = alphabet.get_batch_converter()
            self.model.eval().cuda()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        protein, ligand = self.dataset[item]['atoms_pocket'], self.dataset[item]['atoms_ligand']
        protein_ca = protein[protein['name'] == 'CA']

        df = pd.concat([protein_ca, ligand], ignore_index=True)
        nodes = torch.as_tensor(list(map(_amino_acids, df.resname)), dtype=torch.long, device=self.device)
        coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
        label = torch.as_tensor(self.dataset[item]['scores']['neglog_aff'], device=self.device)
        if self.plm:
            token_reps = get_plm_reps(protein_ca, self.model, self.batch_converter)
            return nodes, coords, label, token_reps
        return nodes, coords, label


class LBAModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.plm:
            _EMBED_DIM = 1280
            weights_freeze = torch.zeros(20, _EMBED_DIM)  # 20 residue types
            weights_train = torch.nn.parameter.Parameter(torch.rand(1, _EMBED_DIM))  # 1 type for ligand
            self.embed = torch.cat((weights_freeze, weights_train), 0).cuda()  # freeze the protein residue embedding

    def forward(self, batch, **kwargs):
        if self.plm:
            h_V = F.embedding(batch.atoms, self.embed) + batch.plm
        else:
            h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)

        batch_id = batch.batch
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        out = self.dense(out).squeeze(-1)
        return out


class LEPTransform(BaseTransform):
    """
    Transforms dict-style entries from the ATOM3D LEP dataset
    to featurized graphs. Returns a tuple (active, inactive) of 
    `torch_geometric.data.Data` graphs with the (same) attribute
    `label` which is equal to 1. if the ligand activates the protein
    and 0. otherwise, and all structural attributes as described
    in BaseTransform.
    
    The transform combines the atomic coordinates of the pocket
    and ligand atoms and treats them as a single structure / graph.
    
    Excludes hydrogen atoms.
    """

    def __call__(self, elem):
        active, inactive = elem['atoms_active'], elem['atoms_inactive']
        with torch.no_grad():
            active, inactive = map(self._to_graph, (active, inactive))
        active.label = inactive.label = 1. if elem['label'] == 'A' else 0.
        return active, inactive

    def _to_graph(self, df):
        df = df[df.element != 'H'].reset_index(drop=True)
        return super().__call__(df)


class LEPModel(BaseModel):
    """
    Extends BaseModel to accept a tuple (batch1, batch2) of `torch_geometric.data.Batch` graphs, where each graph index in a batch is paired with the same graph index in the
    other batch.
    Returns a single scalar for each graph pair which can be used as a logit in binary classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(4 * ns, 1))

    def forward(self, batch):
        out1, out2 = map(self._gnn_forward, batch)
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)

    def _gnn_forward(self, graph):
        return super().forward(graph, dense=False)


class MSPTransform(BaseTransform):
    """
    Transforms dict-style entries from the ATOM3D MSP dataset to featurized graphs. Returns a tuple (original, mutated) of
    `torch_geometric.data.Data` graphs with the (same) attribute`label` which is equal to 1. if the mutation stabilizes the
    complex and 0. otherwise, and all structural attributes as described in BaseTransform.
    The transform combines the atomic coordinates of the two proteis in each complex and treats them as a single structure / graph.
    Adapted from https://github.com/drorlab/atom3d/blob/master/examples/msp/gnn/data.py
    Excludes hydrogen atoms.
    """

    def __call__(self, elem):
        mutation = elem['id'].split('_')[-1]
        orig_df = elem['original_atoms'].reset_index(drop=True)
        mut_df = elem['mutated_atoms'].reset_index(drop=True)
        with torch.no_grad():
            original, mutated = self._transform(orig_df, mutation), self._transform(mut_df, mutation)
        original.label = mutated.label = 1. if elem['label'] == '1' else 0.
        return original, mutated

    def _transform(self, df, mutation):
        df = df[df.element != 'H'].reset_index(drop=True)
        data = super().__call__(df)
        data.node_mask = self._extract_node_mask(df, mutation)
        return data

    def _extract_node_mask(self, df, mutation):
        chain, res = mutation[1], int(mutation[2:-1])
        idx = df.index[(df.chain.values == chain) & (df.residue.values == res)].values
        mask = torch.zeros(len(df), dtype=torch.long, device=self.device)
        mask[idx] = 1
        return mask


class MSPModel(BaseModel):
    """
    Extends BaseModel to accept a tuple (batch1, batch2) of `torch_geometric.data.Batch` graphs, where each graph
    index in a batch is paired with the same graph index in the other batch.
    
    As noted in the manuscript, MSPModel uses the final embeddings averaged over the residue of interest instead of the entire graph.
    
    Returns a single scalar for each graph pair which can be used as a logit in binary classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(2 * ns, 4 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(4 * ns, 1))

    def forward(self, batch):
        out1, out2 = map(self._gnn_forward, batch)
        out = torch.cat([out1, out2], dim=-1)
        out = self.dense(out)
        return torch.sigmoid(out).squeeze(-1)

    def _gnn_forward(self, graph):
        out = super().forward(graph, scatter_mean=False, dense=False)
        out = out * graph.node_mask.unsqueeze(-1)
        out = torch_scatter.scatter_add(out, graph.batch, dim=0)
        count = torch_scatter.scatter_add(graph.node_mask, graph.batch)
        return out / count.unsqueeze(-1)


class RESDataset(IterableDataset):
    """
    A `torch.utils.data.IterableDataset` wrapper around a ATOM3D RES dataset.
    
    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon, 
    and all structural attributes as described in BaseTransform.

    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    """

    def __init__(self, lmdb_dataset, split_path):
        self.dataset = LMDBDataset(lmdb_dataset)
        self.idx = list(map(int, open(split_path).read().split()))
        self.transform = BaseTransform()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))), shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end], shuffle=True)
        return gen

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)  # shuffle data index
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[self.idx[idx]]
                atoms = data['atoms']
                for sub in data['labels'].itertuples():
                    _, num, aa = sub.subunit.split('_')
                    num, aa = int(num), _amino_acids(aa)
                    if aa == 20: continue
                    my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                    ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA'))[0]  # get index of CA
                    if len(ca_idx) != 1: continue

                    with torch.no_grad():
                        graph = self.transform(my_atoms)
                        graph.label = aa
                        graph.ca_idx = int(ca_idx)
                        yield graph


class RESModel(BaseModel):
    """
    Extends BaseModel to output a 20-dim vector instead of a single scalar for each graph, which can be used as logits in 20-way classification.
    As noted in the manuscript, RESModel uses the final alpha carbon embeddings instead of the graph mean embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, 20))

    def forward(self, batch):
        out = super().forward(batch, scatter_mean=False)
        return out[batch.ca_idx + batch.ptr[:-1]]


class TOYDataset(IterableDataset):

    def __init__(self, lmdb_dataset, label='dist', connection='rball'):
        self.dataset = lmdb_dataset
        self.transform = BaseTransform(connection=connection)
        self.label = label

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.dataset))), shuffle=True)
        else:
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(list(range(len(self.dataset)))[iter_start:iter_end], shuffle=True)
        return gen

    def _dataset_generator(self, indices, shuffle=True, level='residue'):
        if shuffle: random.shuffle(indices)  # shuffle data index
        with torch.no_grad():
            for idx in indices:
                df = self.dataset[idx]['atoms']
                if level == 'atom':
                    df = df[df.element != 'H'].reset_index(drop=True)
                    ca_idx = np.where(df.name == 'CA')[0]
                    n_ca = len(ca_idx)
                else:
                    df = df[df['name'] == 'CA']
                    n_ca = len(df)

                with torch.no_grad():
                    graph = self.transform(df)
                    if self.label == 'id':  # label is the position index with a CE loss
                        graph.label = torch.arange(0, n_ca)
                    else:  # label is the minimum distance to the two sides with a MSE loss
                        graph.label = torch.FloatTensor([min(i, n_ca - i) for i in range(0, n_ca)])
                    if level == 'atom': graph.ca_idx = ca_idx
                    yield graph


class TOYDataset2(Dataset):

    def __init__(self, lmdb_dataset, label='dist', device='cpu'):
        self.dataset = lmdb_dataset
        self.label = label
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        with torch.no_grad():
            df = self.dataset[item]['atoms']
            df = df[df['name'] == 'CA']
            nodes = torch.as_tensor(list(map(_amino_acids, df.resname)), dtype=torch.long, device=self.device)
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(), dtype=torch.float32, device=self.device)
            if self.label == 'id':
                label = torch.arange(0, len(df))
            else:
                label = torch.FloatTensor([min(i, len(df) - i) for i in range(0, len(df))])
        return nodes, coords, label


class TOYModel(BaseModel):

    def __init__(self, label='dist', **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        if label == 'id':               # classification task with the maximum number of residue
            self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, _MAX_SEQUENCE_LENGTH))
        elif label == 'dist':           # regression task
            self.dense = nn.Sequential(nn.Linear(ns, 2 * ns), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(2 * ns, 1), nn.ReLU())

    def forward(self, batch, level='residue', **kwargs):
        out = super().forward(batch, scatter_mean=False)
        if level == 'atom':             # ptr corrects the position of ca_idx in batch
            idx = [batch.ca_idx[0]]
            for i in range(1, len(batch.ca_idx)):
                idx.append(batch.ca_idx[i] + batch.ptr[i].item())
            idx = np.concatenate(idx)
            return out[idx]
        else:
            return out
