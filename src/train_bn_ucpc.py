import sys
from os import path

import torch
import torchvision
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from packages.spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from packages.spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from packages.spn.algorithms.layerwise.distributions import Bernoulli, Categorical
from constraint.constraints import *
import time
import argparse
from tqdm import trange
from pathlib import Path

from datasets import get_bn_dataset, get_data_loader


def make_spn(S, I, R, D, F, C, device, leaf_base_class, leaf_base_kwargs=None) -> RatSpn:
        """Construct the RatSpn"""

        # Setup RatSpnConfig
        config = RatSpnConfig()
        config.F = F
        config.R = R
        config.D = D
        config.I = I
        config.S = S
        config.C = C
        config.dropout = 0.0
        config.leaf_base_class = leaf_base_class 
        config.leaf_base_kwargs = {} if leaf_base_kwargs is None else leaf_base_kwargs

        # Construct RatSpn from config
        model = RatSpn(config)

        model = model.to(device)
        model.train()

        print("Using device:", device)
        return model


def train_model(model, train_loader, constraints, n_iterations=1000, t_max=0, tol=1e-4, device='cpu'):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    prev_loss, total_loss = 0, 0
    rel_change_loss, rel_change_penalty = 0, 0
    prev_penalty, total_penalty = 0, 0
    config_data = {}
    prev_loss = 0
    for t in range(-1, t_max):
        torch.cuda.empty_cache()
        total_data_loss, total_penalty = 0, 0
        for iteration in trange(n_iterations):
            total_loss = 0
            for (data,) in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = -outputs.sum() / data.shape[0]
                total_data_loss += loss.detach()
                
                penalty = torch.tensor(0.0, device=device)
                for constraint in constraints:
                    penalty += constraint.violation(model, train_loader.dataset, config_data, device=device, batch_size=64)
                
                total_penalty += penalty.detach()
                if t >= 0:
                    lambda_ = 10**t
                    loss += lambda_*penalty
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
            
            
            if iteration > 0:
                rel_change_loss = (prev_loss - total_loss) / prev_loss
                
            if iteration >0 and rel_change_loss < tol:
                break
            prev_loss = total_loss
        
        if t >= 0 and total_penalty/(iteration+1) < tol:
            break
        print (f"{iteration} {t} {total_data_loss/(iteration+1):.4f} {rel_change_loss} {total_penalty/(iteration+1)}")
        
    return model
            

import random
import numpy as np


name = sys.argv[1]
trial = int(sys.argv[2])
random.seed(trial)
np.random.seed(trial)
torch.manual_seed(trial)

r, names, train, test, cis = get_bn_dataset(name)
train_dataset, train_loader = get_data_loader(train, batch_size=64)
test_dataset, test_loader = get_data_loader(train, batch_size=64)

rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 20, 20, 2, 5, 1,Categorical #RatNormal
n_features = len(r)

device = torch.device("cuda")
dropout = 0

constraints = [
    ContextSpecificIndependence(names.index(X), names.index(Y), names.index(Z), z, r)
    for X, Y, Z in cis[0:2]
    for z in range(r[names.index(Z)])
] + [
    ContextSpecificIndependence(names.index(Y), names.index(X), names.index(Z), z, r)
    for X, Y, Z in cis[0:2]
    for z in range(r[names.index(Z)])
]

model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, F=n_features, C=rat_C,leaf_base_class=leaves, leaf_base_kwargs=dict(num_bins=max(r)))

model = train_model(model, train_loader, constraints, t_max=0, tol=1e-16,device=device)
torch.save(model, f"{name}-{trial}-0.pt")

model = train_model(model, train_loader, constraints, t_max=10, tol=1e-16,device=device)
torch.save(model, f"{name}-{trial}-10.pt")
