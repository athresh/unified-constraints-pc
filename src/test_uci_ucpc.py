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

from datasets import get_uci_dataset, get_data_loader

import random
import numpy as np


# random.seed(trial)
# np.random.seed(trial)
# torch.manual_seed(trial)

for name in ["breast-cancer", "diabetes", "thyroid",  "heart-disease", "numom2b"]: # , , ,
    df, r, target, monotonicities = get_uci_dataset(name)
    names = df.columns.tolist()
    train, test = train_test_split(df, stratify=df[target], test_size=0.5, random_state=0)
    
    train_dataset, train_loader = get_data_loader(train, batch_size=64)
    test_dataset, test_loader = get_data_loader(test, batch_size=64)
    
    base = []
    ll = []
    for trial in range(3):
        model = torch.load(f"{name}-{trial}-{0}.pt")
        base.append(log_likelihood(test_loader, model, device="cpu"))
        
        model = torch.load(f"{name}-{trial}-{10}.pt")
        ll.append(log_likelihood(test_loader, model, device="cpu"))
    
    print (f"{name}, {np.mean(base):.3f} ± {np.std(base):.2f}, {np.mean(ll):.3f} ±{np.std(ll):.2f}")