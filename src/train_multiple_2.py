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


def train_model(model, train_loader, constraints, n_iterations=1000, t_min=-1, t_max=0, tol=1e-4, device='cpu'):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    prev_loss, total_loss = 0, 0
    rel_change_loss, rel_change_penalty = 0, 0
    prev_penalty, total_penalty = 0, 0
    config_data = {}
    prev_loss = 0
    for t in range(t_min, t_max):
        # torch.cuda.empty_cache()
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
                
            # if iteration > 0 and rel_change_loss < tol:
            #    break
            prev_loss = total_loss
        
        if t >= 0 and total_penalty/(iteration+1) < tol:
            break
        print (f"{iteration} {t} {total_data_loss/(iteration+1):.4f} {rel_change_loss} {total_penalty/(iteration+1)}")
        
    return model
            

def train_model(model, train_loader, constraints, n_iterations=1000, t_min=-1, t_max=0, tol=1e-4, device='cpu'):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    prev_loss, total_loss = 0, 0
    rel_change_loss, rel_change_penalty = 0, 0
    prev_penalty, total_penalty = 0, 0
    config_data = {}
    prev_loss = 0
    for t in range(t_min, t_max):
        # torch.cuda.empty_cache()
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
                
            """
            if iteration >0 and rel_change_loss < tol:
                break
            """
            prev_loss = total_loss

        with torch.no_grad():
            penalty = torch.tensor(0.0, device=device)
            for constraint in constraints:
                penalty += constraint.violation(model, train_loader.dataset, config_data, device=device, batch_size=64)
        total_penalty = float(penalty)
        
        
        print (f"{iteration} {t} {total_data_loss/(iteration+1):.4f} {rel_change_loss} {total_penalty/(iteration+1)}")
        if t >= 0 and total_penalty/(iteration+1) < tol:
            break
    return model

import random
import numpy as np


name = "numom2b_multi_apo"
trial = int(sys.argv[1])
random.seed(trial)
np.random.seed(trial)
torch.manual_seed(trial)

r, names, train, test, cis = get_bn_dataset(name)
train_dataset, train_loader = get_data_loader(train, batch_size=512)
test_dataset, test_loader = get_data_loader(train, batch_size=512)

rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 20, 20, 2, 5, 1, Categorical #RatNormal
n_features = len(r)

device = torch.device("cpu")
dropout = 0

constraints = []

model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, F=n_features, C=rat_C,leaf_base_class=leaves, leaf_base_kwargs=dict(num_bins=max(r)))
model = train_model(model, train_loader, constraints, t_max=0, tol=1e-8,device=device, n_iterations=100)
torch.save(model, f"{name}-multiple-{trial}-0.pt")


constraints = [
    ContextSpecificIndependence(names.index(X), names.index(Y), names.index(Z), z, r)
    for X, Y, Z in cis[0:1]
    for z in range(r[names.index(Z)])
] 

model = train_model(model, train_loader, constraints, t_min=0, t_max=10, tol=1e-8,device=device, n_iterations=100)
torch.save(model, f"{name}-multiple-{trial}-10.pt")

if name == "asia":
    parent1 = "either"
    target1 = "xray"
    sign = +1
    epsilon = 1e-2
    constraints += [MonotonicityConstraint(names.index(parent1), names.index(target1), r, sign, epsilon)]
elif name == "earthquake":
    parent1 = "Alarm"
    target1 = "MaryCalls"
    sign = +1
    epsilon = 1e-2
    constraints += [MonotonicityConstraint(names.index(parent1), names.index(target1), r, sign, epsilon)]
elif name == "numom2b_multi_apo":
    parent1 = "BMI"
    target1 = "PTB"
    sign = +1
    epsilon = 1e-3
    constraints += [MonotonicityConstraint(names.index(parent1), names.index(target1), r, sign, epsilon)]


model = train_model(model, train_loader, constraints, t_min=0, t_max=10, tol=1e-8,device=device, n_iterations=100)
torch.save(model, f"{name}-multiple-{trial}-20.pt")