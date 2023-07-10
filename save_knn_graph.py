import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from accelerate import Accelerator

import os, sys
import pandas as pd
import time
import types 
import argparse 
import numpy as np

from features_data_loader import FeaturesDataset, CSADataset, CSADatasetK
from models import * 
from utils import load_trained_ssa_layers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--jobid', type=int, default=0)
parser.add_argument('--logs_dir', type=str, default='logs/csa_n_heads_1_K_1/Bed')
parser.add_argument('--ssa_logs_dir', type=str, default='logs/ssa_n_heads_1/Bed')

parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--testing', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--partname', type=str, default='Bed')
parser.add_argument('--train_iters', type=int, default=3000)
parser.add_argument('--num_classes', type=int, default=15)
parser.add_argument('--attention_type', type=str, default='csa')
parser.add_argument('--K', type=int, default=1)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

big_classes = ['Chair', 'Lamp', 'StorageFurniture', 'Table']

args = parser.parse_args() 

def gpu_mem():
    t = torch.cuda.get_device_properties(0).total_memory * 1e-9
    r = torch.cuda.memory_reserved(0) * 1e-9
    a = torch.cuda.memory_allocated(0) * 1e-9

def createdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created!')

def logprint(log, logs_fp=None):
    if not isinstance(logs_fp, type(None)):
        with open(logs_fp, 'a') as f:
            f.write(log + '\n')
    print(log)

def torch_save(obj, name, path):
    torch.save({name:obj}, path)
    print(f'{path} saved!')

def torch_load(name, path):
    ckpt = torch.load(path)
    print('model loaded from', path)
    return ckpt[name]

def update_knn_graphs(train_root, test_root, model, train_loader, test_loader, K, logs_dir):
    
    if args.partname in big_classes:
        candidate_shape_indices = model.get_center_shape_indices(train_loader) 

        train_knn_cand_shape_indices = model.get_knn_graph_big(train_loader, train_loader, candidate_shape_indices, K).cpu().numpy()
        train_knn_graph = []
        for i in range(len(train_knn_cand_shape_indices)):
            knn = []
            for j in range(len(train_knn_cand_shape_indices[i])):
                knn += [candidate_shape_indices[train_knn_cand_shape_indices[i][j]]]
            train_knn_graph += [knn.copy()]

        test_knn_cand_shape_indices = model.get_knn_graph_big(test_loader, train_loader, candidate_shape_indices, K).cpu().numpy()
        test_knn_graph = []
        for i in range(len(test_knn_cand_shape_indices)):
            knn = []
            for j in range(len(test_knn_cand_shape_indices[i])):
                knn += [candidate_shape_indices[test_knn_cand_shape_indices[i][j]]]
            test_knn_graph += [knn.copy()]
        
        train_knn_graph = np.array(train_knn_graph)
        test_knn_graph = np.array(test_knn_graph)
    else:
        train_ssa_feats = model.get_all_feats(logs_dir, train_loader, K, 'train')
        train_ssa_feats = train_ssa_feats.to(device)
        train_knn_graph = model.get_knn_graph(train_ssa_feats, train_ssa_feats, K).cpu().numpy()
        train_ssa_feats = train_ssa_feats.cpu()

        test_ssa_feats = model.get_all_feats(logs_dir, test_loader, K, 'test')
        train_ssa_feats = train_ssa_feats.to(device)
        test_ssa_feats = test_ssa_feats.to(device)
        test_knn_graph = model.get_knn_graph(test_ssa_feats, train_ssa_feats, K).cpu().numpy()

        train_ssa_feats = train_ssa_feats.cpu()
        test_ssa_feats = test_ssa_feats.cpu()
        train_ssa_feats = None 
        test_ssa_feats = None

    return train_knn_graph, test_knn_graph

partname = args.partname
create_dirs(f'logs/knn_graphs/n_heads_{args.n_heads}/{args.partname}')

dataroot = '/work/siddhantgarg_umass_edu/int-vis/O-CNN/tensorflow/script/logs/partnet_weights_and_logs/partnet_finetune/{}_data_features/{}' # change to your data root

train_root = dataroot.format('train', args.partname)
test_root = dataroot.format('test', args.partname)

train_dataset = FeaturesDataset(dataroot.format('train', args.partname), args.attention_type)
train_loader = DataLoader(train_dataset, 1, shuffle=False, num_workers=args.num_workers)

test_dataset = FeaturesDataset(dataroot.format('test', args.partname), args.attention_type)
test_loader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers)

ssa_logs_dir = f'logs/ssa_n_heads_{args.n_heads}/run_1/{args.partname}'
model = get_model(args.attention_type, args.num_classes, args.n_heads, args.K).to(device)
model = load_trained_ssa_layers(model, ssa_logs_dir)
logprint('trained_ssa_layers imported!', 'logs/knn_graphs/logs.txt')

train_knn_graph, test_knn_graph = update_knn_graphs(train_root, test_root, model, train_loader, test_loader, 10, 'logs/knn_graphs')

trainp = f'logs/knn_graphs/n_heads_{args.n_heads}/{args.partname}/train.npy'
testp = f'logs/knn_graphs/n_heads_{args.n_heads}/{args.partname}/test.npy'

with open(trainp, 'wb') as f:
    np.save(f, train_knn_graph)
    print(f'saved to {trainp}')

with open(testp, 'wb') as f:
    np.save(f, test_knn_graph)
    print(f'saved to {testp}')
