import faulthandler
faulthandler.enable()

import argparse
from collections import defaultdict
from functools import partial

import numpy as np
import os
import sklearn.metrics as sk_metrics
import time
import torch
import torch.nn as nn
import torch_geometric
import tqdm
from atom3d.datasets import LMDBDataset
from atom3d.splits.splits import split_randomly
from atom3d.util import metrics
from torch.nn.utils.rnn import pad_sequence
from types import SimpleNamespace

import gvp
import gvp.atom3d
from gvp import set_seed, Logger
from egnn import egnn_clean as eg

print = partial(print, flush=True)
models_dir = 'models'

parser = argparse.ArgumentParser()
parser.add_argument('task', metavar='TASK', default='PSR', choices=['PSR', 'PPI', 'RES', 'MSP', 'LBA', 'LEP', 'TOY'])
parser.add_argument('--toy', metavar='TOY TARGET', type=str, default='id', choices=['id', 'dist'])
parser.add_argument('--connect', metavar='CONNECTION', type=str, default='rball', choices=['rball', 'knn'])
parser.add_argument('--model', metavar='MODEL', type=str, default='gvp', choices=['egnn', 'gvp', 'molformer'])  # metavar will show in help information
parser.add_argument('--plm', metavar='PLM', type=int, default=0, help='whether use PLM features')
parser.add_argument('--num-workers', metavar='N', type=int, default=4, help='number of threads for loading data, default=4')
parser.add_argument('--lba-split', metavar='SPLIT', type=int, choices=[30, 60], help='identity cutoff for LBA, 30 (default) or 60', default=30)
parser.add_argument('--batch', metavar='SIZE', type=int, default=32, help='batch size, default=32 for gvp-gnn')
parser.add_argument('--train-time', metavar='MINUTES', type=int, default=120, help='maximum time between evaluations on valset, default=120 minutes')
parser.add_argument('--val-time', metavar='MINUTES', type=int, default=20, help='maximum time per evaluation on valset, default=20 minutes')
parser.add_argument('--epochs', metavar='N', type=int, default=200, help='training epochs, default=50')
parser.add_argument('--test', metavar='PATH', default=None, help='evaluate a trained model')
parser.add_argument('--lr', metavar='RATE', default=1e-4, type=float, help='learning rate')
parser.add_argument('--load', metavar='PATH', default=None, help='initialize first 2 GNN layers with pretrained weights')
parser.add_argument('--gpu', metavar='GPU', type=str, default='0')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()
task_tag = args.task + str(args.lba_split) if args.task == 'LBA' else args.task
log = Logger(f'./', f'training_{task_tag}_{args.model}_{args.plm}.log')
device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and not args.debug else 'cpu'
set_seed(0)


def collate(samples):
    if args.plm:
        nodes, coords, label, token_reps = zip(*samples)
    else:
        nodes, coords, label = zip(*samples)
    if args.task == 'PPI':
        nodes1, nodes2 = zip(*nodes)
        coords1, coords2 = zip(*coords)
        label1, label2 = zip(*label)
        nodes = nodes1 + nodes2
        coords = coords1 + coords2
        label = label1 + label2
        if args.plm:
            token_reps1, token_reps2 = zip(*token_reps)
            token_reps = token_reps1 + token_reps2
    nodes = pad_sequence(nodes, batch_first=True, padding_value=21)         # 21 is the token id of [UNK]
    coords = pad_sequence(coords, batch_first=True, padding_value=0.0)
    if args.task in ['TOY', 'PPI']:
        label = pad_sequence(label, batch_first=True, padding_value=-1)
    elif args.task == 'PSR':
        label, id = zip(*label)
        label = torch.stack(label)
    else:
        label = torch.stack(label)
    batch = SimpleNamespace(label=label, nodes=nodes, coords=coords)
    if args.plm:
        token_reps = pad_sequence(token_reps, batch_first=True, padding_value=0.0)
        batch.token_reps = token_reps
    if args.task == 'PSR':
        batch.id = id
    return batch


def main():
    if args.model == 'molformer': args.connect = 'FC'
    datasets = get_datasets(args.task, args.lba_split)
    if args.plm: args.num_workers = 0  # https://stackoverflow.com/questions/59081290/not-using-multiprocessing-but-get-cuda-error-on-google-colab-while-using-pytorch
    if args.model == 'molformer':      # https://github.com/pytorch/pytorch/issues/40403
        if args.task == 'TOY':
            args.batch = 4
        dataloader = partial(torch.utils.data.DataLoader, num_workers=args.num_workers, batch_size=args.batch, collate_fn=collate)
    else:                              # older version is data.Dataloader
        dataloader = partial(torch_geometric.loader.DataLoader, num_workers=args.num_workers, batch_size=args.batch)
    if args.task not in ['PPI', 'RES', 'TOY']: dataloader = partial(dataloader, shuffle=True)
    log.logger.info(f'{"=" * 40} Configuration {"=" * 40}\nModel: {args.model}; Task: {args.task}; PLM: {args.plm}; Graph: {args.connect}; Epochs: {args.epochs};'
                    f' Batch Szie: {args.batch}; GPU: {args.gpu}; Worker: {args.num_workers}\n{"=" * 40} Start Training {"=" * 40}')

    trainset, valset, testset = map(dataloader, datasets)
    model = get_model(args.task, args.model).to(device)
    if args.test:
        test(model, testset, args.test)
    else:
        model_path = train(model, trainset, valset, patience=8)
        test(model, testset, model_path.split('/')[-1])


def test(model, testset, model_path):
    model.load_state_dict(torch.load('models/' + model_path))
    print('Loading model weight successfully! Start to test. ')
    model.eval()
    t = tqdm.tqdm(testset)
    metrics = get_metrics(args.task)
    targets, predicts, ids = [], [], []
    with torch.no_grad():
        for batch in t:
            pred = forward(model, batch, device)
            label = get_label(batch)
            if args.model == 'molformer' and args.task in ['TOY', 'PPI']:
                mask = (batch.nodes != 21)
                label, pred = label[mask], pred[mask]

            if args.task == 'RES' or (args.task == 'TOY' and args.toy == 'id'): pred = pred.argmax(dim=-1)
            if args.task == 'PSR': ids.extend(batch.id)
            targets.extend(list(label.cpu().numpy()))
            predicts.extend(list(pred.cpu().numpy()))

    for name, func in metrics.items():
        if args.task == 'PSR': func = partial(func, ids=ids)
        value = func(targets, predicts)
        log.logger.info(f"{name}: {value}")


def train(model, trainset, valset, patience=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5, min_lr=5e-7)

    best_path, best_val, wait = None, np.inf, 0
    if not os.path.exists(models_dir): os.makedirs(models_dir)
    for epoch in range(args.epochs):
        model.train()
        train_loss = loop(trainset, model, optimizer=optimizer, max_time=args.train_time)
        path = f"{models_dir}/{args.task}_{args.model}_plm{args.plm}_epoch{epoch}_{float(time.time())}.pt"
        torch.save(model.state_dict(), path)
        model.eval()
        with torch.no_grad():
            val_loss = loop(valset, model, max_time=args.val_time)
        log.logger.info(f'[Epoch {epoch}] Train loss: {train_loss:.8f} Val loss: {val_loss:.8f}')
        if val_loss < best_val:
            best_path, best_val = path, val_loss
        else:
            wait += 1
        log.logger.info(f'Best {best_path} Val loss: {best_val:.8f}\n')
        if wait >= patience: break                                          # early stop
        lr_scheduler.step(val_loss)                                         # based on validation loss
    return best_path


def loop(dataset, model, optimizer=None, max_time=None):
    start = time.time()
    loss_fn = get_loss(args.task)
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0

    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            out = forward(model, batch, device)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise e
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        label = get_label(batch)
        if args.model == 'molformer' and args.task in ['TOY', 'PPI']:
            mask = (batch.nodes != 21)
            label, out = label[mask], out[mask]
        loss_value = loss_fn(out, label.cuda())
        total_loss += float(loss_value)
        total_count += 1

        if optimizer:
            try:
                loss_value.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise e
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
        t.set_description(f"Loss: {total_loss / total_count:.5f}")  # tdqm的description

    return total_loss / total_count


def load(model, path):
    params = torch.load(path)
    state_dict = model.state_dict()
    for name, p in params.items():
        if name in state_dict and name[:8] in ['layers.0', 'layers.1'] and state_dict[name].shape == p.shape:
            print("Loading", name)
            model.state_dict()[name].copy_(p)


def get_label(batch):
    if type(batch) in [list, tuple]:
        return torch.cat([i.label for i in batch])
    return batch.label


def get_metrics(task):
    def _correlation(metric, targets, predict, ids=None, glob=True):
        if glob: return metric(targets, predict)
        _targets, _predict = defaultdict(list), defaultdict(list)
        for _t, _p, _id in zip(targets, predict, ids):
            _targets[_id].append(_t)
            _predict[_id].append(_p)
        return np.mean([metric(_targets[_id], _predict[_id]) for _id in _targets])

    correlations = {'pearson': partial(_correlation, metrics.pearson), 'kendall': partial(_correlation, metrics.kendall), 'spearman': partial(_correlation, metrics.spearman)}
    mean_correlations = {f'mean {k}': partial(v, glob=False) for k, v in correlations.items()}

    if task == 'TOY':
        return {'rmse': partial(sk_metrics.mean_squared_error, squared=False)} if args.toy == 'dist' else {'accuracy': metrics.accuracy}
    else:
        return {'PSR': {**correlations, **mean_correlations}, 'PPI': {'auroc': metrics.auroc}, 'RES': {'accuracy': metrics.accuracy},
                'MSP': {'auroc': metrics.auroc, 'auprc': metrics.auprc}, 'LEP': {'auroc': metrics.auroc, 'auprc': metrics.auprc},
                'LBA': {**correlations, 'rmse': partial(sk_metrics.mean_squared_error, squared=False)}}[task]


def get_loss(task):
    if task in ['PSR', 'LBA']:
        return nn.MSELoss()                      # regression
    elif task in ['PPI', 'MSP', 'LEP']:
        return nn.BCELoss()                      # binary classification
    elif task in ['RES']:
        return nn.CrossEntropyLoss()             # multiclass classification
    elif task in ['TOY']:
        return nn.MSELoss() if args.toy == 'dist' else nn.CrossEntropyLoss()


def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = [x.to(device) for x in batch]      # PPI two graphs
    elif type(batch) == SimpleNamespace:
        for k in batch.__dict__:
            if k != 'id':
                batch.__dict__[k] = batch.__dict__[k].to(device)
    else:
        batch = batch.to(device)
    return model(batch)


def get_datasets(task, lba_split=30):
    data_path = {'RES': 'atom3d-data/RES/raw/RES/data/', 'PPI': 'data/PPI/DIPS-split/data/', 'PSR': 'data/PSR/split-by-year/data/',
                 'MSP': 'atom3d-data/MSP/splits/split-by-sequence-identity-30/data/', 'LEP': 'atom3d-data/LEP/splits/split-by-protein/data/',
                 'LBA': f'data/LBA/split-by-sequence-identity-{lba_split}/data/', 'TOY': 'data/TOY/split-by-cath-topology/data/'}[task]      # TOY use the test dataset of RES

    if task == 'RES':
        split_path = 'atom3d-data/RES/splits/split-by-cath-topology/indices/'
        dataset = partial(gvp.atom3d.RESDataset, data_path)
        trainset = dataset(split_path=split_path + 'train_indices.txt')
        valset = dataset(split_path=split_path + 'val_indices.txt')
        testset = dataset(split_path=split_path + 'test_indices.txt')
    elif task == 'PPI':
        if args.model == 'molformer':
            train_dataset, val_dataset, test_dataset = split_randomly(LMDBDataset(data_path + 'test'))
            trainset = gvp.atom3d.PPIDataset(train_dataset, plm=args.plm)
            valset = gvp.atom3d.PPIDataset(val_dataset, plm=args.plm)
            testset = gvp.atom3d.PPIDataset(test_dataset, plm=args.plm)
        else:
            dataset = LMDBDataset(data_path + 'test', transform=gvp.atom3d.PPITransform(plm=args.plm))
            trainset, valset, testset = split_randomly(dataset)
    elif task == 'TOY':
        train_dataset, val_dataset, test_dataset = split_randomly(LMDBDataset(data_path + 'test'))
        if args.model == 'molformer':
            trainset = gvp.atom3d.TOYDataset2(train_dataset, label=args.toy)
            valset = gvp.atom3d.TOYDataset2(val_dataset, label=args.toy)
            testset = gvp.atom3d.TOYDataset2(test_dataset, label=args.toy)
        else:
            trainset = gvp.atom3d.TOYDataset(train_dataset, label=args.toy, connection=args.connect)
            valset = gvp.atom3d.TOYDataset(val_dataset, label=args.toy, connection=args.connect)
            testset = gvp.atom3d.TOYDataset(test_dataset, label=args.toy, connection=args.connect)
    else:
        if task == 'PSR':
            if args.model == 'molformer':
                trainset = gvp.atom3d.PSRDataset(LMDBDataset(data_path + 'train'), plm=args.plm)
                valset = gvp.atom3d.PSRDataset(LMDBDataset(data_path + 'val'), plm=args.plm)
                testset = gvp.atom3d.PSRDataset(LMDBDataset(data_path + 'test'), plm=args.plm)
                return trainset, valset, testset
            transform = gvp.atom3d.PSRTransform(plm=args.plm)
        elif task == 'LBA':
            if args.model == 'molformer':
                trainset = gvp.atom3d.LBADataset(LMDBDataset(data_path + 'train'), plm=args.plm)
                valset = gvp.atom3d.LBADataset(LMDBDataset(data_path + 'val'), plm=args.plm)
                testset = gvp.atom3d.LBADataset(LMDBDataset(data_path + 'test'), plm=args.plm)
                return trainset, valset, testset
            transform = gvp.atom3d.LBATransform(plm=args.plm)
        else:
            transform = {'MSP': gvp.atom3d.MSPTransform, 'LEP': gvp.atom3d.LEPTransform}[task]()
        trainset = LMDBDataset(data_path + 'train', transform=transform)
        valset = LMDBDataset(data_path + 'val', transform=transform)
        testset = LMDBDataset(data_path + 'test', transform=transform)
        print(len(trainset), len(valset), len(testset))
    return trainset, valset, testset


def get_model(task, model):
    if model == 'gvp':
        if task == 'TOY':
            return gvp.atom3d.TOYModel(args.toy)
        elif task == 'PSR':
            return gvp.atom3d.BaseModel(plm=args.plm)
        elif task == 'LBA':
            return gvp.atom3d.LBAModel(plm=args.plm)
        elif task == 'PPI':
            return gvp.atom3d.PPIModel(plm=args.plm)
        return {'RES': gvp.atom3d.RESModel, 'MSP': gvp.atom3d.MSPModel, 'LEP': gvp.atom3d.LEPModel}[task]()
    elif model == 'egnn':
        if task == 'TOY':
            return eg.TOYModel(args.toy)
        elif task == 'PSR':
            return eg.PSRModel(plm=args.plm)
        elif task == 'LBA':
            return eg.LBAModel(plm=args.plm)
        elif task == 'PPI':
            return eg.PPIModel(plm=args.plm)
        return {}[task]()
    elif model == 'molformer':
        import molformer.tr_cpe as tr
        if task == 'TOY':
            return tr.TOYModel(args.toy)
        elif task == 'PSR':
            return tr.PSRModel(plm=args.plm)
        elif task == 'LBA':
            return tr.LBAModel(plm=args.plm)
        elif task == 'PPI':
            return tr.PPIModel(plm=args.plm)
        return {}[task]()


if __name__ == "__main__":
    main()
