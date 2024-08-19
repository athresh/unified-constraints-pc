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

import random
import numpy as np


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

for name in ["asia", "earthquake", "sachs", "survey"]:
    r, names, train, test, cis = get_bn_dataset(name)
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