import sys
from os import path
sys.path.append(path.join("..", "src"))

import numpy as np
import pandas as pd
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
from packages.spn.algorithms.layerwise.distributions import Bernoulli
from utils.datasets import gen_dataset
from utils.config_utils import load_config_data
from utils.utils import visualize_3d
from utils.selectors import get_sim_dataloader
from constraint.constraints import GeneralizationConstraint, EqualityConstraint, AbstractConstraint
import time
import argparse
from tqdm import tqdm
from pathlib import Path

def make_spn(S, I, R, D, F, C, device, leaf_base_class) -> RatSpn:
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
        config.leaf_base_kwargs = {}

        # Construct RatSpn from config
        model = RatSpn(config)

        model = model.to(device)
        model.train()

        print("Using device:", device)
        return model


def get_adult_loaders(use_cuda, batch_size):
    # df = pd.read_csv(path.join("..","data","Adult","train_0.csv")).astype(float)
    # df = pd.read_csv(path.join("..","data","Adult","adult.data")).astype(float)
    # for col in df.columns:
    #     if df[col].nunique() > 2:
    #         d  = KBinsDiscretizer(n_bins=2, encode='ordinal',strategy='kmeans', subsample=None)
    #         df[col] = d.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten().astype(int)
    # print (df.nunique())
    # kwargs = {"num_workers": 8, "pin_memory": True} if use_cuda else {}

    test_batch_size = batch_size

    # train, test = train_test_split(df, stratify=df.income, random_state=0)
    train, test = torch.randint(0, 1, size=(1000, 87)).long(), torch.randint(0, 1, size=(1000, 87)).long()
    train_dataset, test_dataset = TensorDataset(torch.Tensor(train)), TensorDataset(torch.Tensor(test))
    # Train data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_dataset, test_dataset, train_loader, test_loader

train_dataset, test_dataset, train_loader, test_loader = get_adult_loaders(False, 32)
rat_S, rat_I, rat_D, rat_R, rat_C, leaves = 20, 20, 5, 5, 1, Bernoulli #RatNormal
n_features = train_loader.dataset[0][0].shape[0]
device=torch.device("cuda")
dropout=0
model = make_spn(S=rat_S, I=rat_I, D=rat_D, R=rat_R, device=device, F=n_features, C=rat_C,leaf_base_class=leaves)

def predict_proba(model, r, data, target_index, marg_indices=None, device='cpu'):
    log_p = torch.zeros((len(data), r[target_index]), device=device)
    log_denom = model(data, (target_index,)) if marg_indices is None else model(data, (target_index,*marg_indices))
    log_denom = log_denom.ravel()
    for i in range(r[target_index]):
        data_i = data.clone()
        data_i[:, target_index] = i
        log_numer = model(data_i) if marg_indices is None else model(data_i, marg_indices)
        log_p[:, i] = log_numer.ravel() - log_denom
    
    return torch.exp(log_p)

class ContextSpecificIndependence(EqualityConstraint):
    def __init__(self, X, Y, Z, z, r):
        # X \indep Y | Z = z
        self.X = X
        self.Y = Y
        self.Z = Z
        self.z = z
        self.r = r
        super().__init__()
    
    def violation(model, dataset, config_data, device="cpu", **kwargs):
        # P(X | Y, Z = z) = P(X | Z = z) 
        
        data = torch.zeros((self.r[self.Y], n_features), device=device)
        for i in  self.r[self.Y]:
            data[i, self.Y] = i
            data[i, self.Z] = z
        
        marg_indices = [i for i in range(n_features) if i not in (self.X, self.Y, self.Z )]
        p1 = predict_proba(model, self.r, data, self.X, marg_indices, device)
        p2 = predict_proba(model, self.r, data, self.X, marg_indices + [self.Y], device)
        delta = self.delta(p1,p2)
        violation = self.degree_violation(delta)
        return violation / (self.r[self.X]*self.r[self.Y])
            
class InequalityConstraint(AbstractConstraint):
    def __init__(self, sign, epsilon):
        super().__init__()
        self.sign = sign
        self.epsilon = epsilon
        
    def delta(self, output_1, output_2):
        delta = torch.sub(output_1, output_2)*self.sign + self.epsilon
        return delta
    def degree_violation(self, delta):
        return torch.sum(torch.max(delta, torch.tensor(0, device=device))**2)

class MonotonicityConstraint(InequalityConstraint):
    def __init__(self, Xj, Xi, r, sign, epsilon):
        super().__init__(sign, epsilon)
        self.Xj = Xj
        self.Xi = Xi
        self.r = r
        
    def violation(self, model, dataset, config_data, device="cpu", **kwargs):
        n_features = len(r)
        marg_indices = [i for i in range(n_features) if i not in (self.Xi, self.Xj)]
        data = torch.zeros((self.r[self.Xj], n_features), device=device)
        for i in range(self.r[self.Xj]):
            data[i, self.Xj] = i
        cdf = torch.cumsum(predict_proba(model, self.r, data, Xj, marg_indices), axis=1)
        total = torch.tensor(0, device=device)
        count = 0
        for xi in range(self.r[self.Xi]):
            for xj_ in range(self.r[self.Xj]):
                for xj in range(xj_):
                    # xj_ > xj
                    delta = self.delta(cdf[xj_, xi], cdf[xj, xi])
                    total += self.degree_violation(delta)
                    count += 1
                    
        return torch.div(total, count)

class FalsePositiveConstraint(InequalityConstraint):
    def __init__(self, target, r, epsilon):
        super().__init__(+1, epsilon)
        self.target = target
        self.r = r
        assert self.r[self.target] == 2
    
    def violation(model, dataset, config_data, device="cpu", batch_size=64, **kwargs):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )
        total = torch.tensor(0, device=device)
        count = 0
        for data in dataloader:
            
            data = data.to(device)
            y = data[:, self.target].clone()
            p = predict_proba(model, data[y == 1], target_index=self.target)
            
            p0 = p[:, 0]
            delta = self.delta(p0, 0.5)
            total += self.degree_violation(delta)
            count += 1
            
        return torch.div(total, count)


def train(model, train_loader, constraints, iterations=100, t_max=0, tol=1e-3, device='cpu'):
    
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    prev_loss, total_loss = 0, 0
    prev_penalty, total_penalty = 0, 0
    t = 0
    zeta = 1
    for iteration in range(100):    
        total_loss, total_penalty = 0, 0
        for (data,) in tqdm(train_loader, total = len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data.float())
            loss = -outputs.sum() / data.shape[0]
            total_loss += loss
            penalty = torch.tensor(0, device=device)
            if t > 0:
                for constraint in constaints:
                    penalty += constraint.violation(model, train_loader.dataset, config_data, device=device, batch_size=64)
            
            total_penalty += penalty
            lambda_ = 10**t
            loss += lambda_*zeta
            
            loss.backward()
            optimizer.step()
            
        if iteration > 0:
            rel_change_loss = (prev_loss - total_loss) / prev_loss
            rel_change_penalty = (prev_penalty - total_penalty) / prev_penalty
            if rel_change_loss < tol:
                if total_penalty > tol**2:
                    t = min(t + 1, t_max)
                    
                    if t == t_max:
                        break
        
        
        if iteration % 3 == 0:
            print (f"{total_loss:.4f}, {rel_change_loss:.4f}, {total_penalty:.4f}, {rel_change_penalty:.4f}")
        prev_loss = total_loss 
        prev_penalty = total_penalty
        return model
            
constraints = [
    FalsePositiveConstraint(86, [2 for _ in range(87)], 0.01)
]

model = train(model.to(device), train_loader, constraints, t_max=0, device=torch.device("cuda"))