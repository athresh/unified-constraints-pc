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
from pgmpy.readwrite import BIFReader
from packages.spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from packages.spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from packages.spn.algorithms.layerwise.distributions import Bernoulli, Categorical
from constraint.constraints import *
import time
import argparse
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
import numpy as np

base = "data/"
def get_uci_dataset(name: str):
    if name == "breast-cancer":
        path = os.path.join(base, "breast-cancer.data")
        columns = ["recurrence", "age", "menopause", "tumor_size", "inv_nodes",
        "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"]
        df = pd.read_csv(path, na_values="?", names=columns).dropna()
        df.recurrence = (df.recurrence == "recurrence-events")
        df.age.replace({
            "10-19":0, "20-29":0, "30-39":0,
            "40-49":1, "50-59":1, "60-69":1,
            "70-79":2, "80-89":2, "90-99":2 
        }, inplace=True)
        df.menopause = (df.menopause != "premeno")
        df.tumor_size.replace({
            "0-4":0, "5-9":0, "10-14":0, 
            "15-19":1, "20-24":1, "25-29":1, 
            "30-34":2, "35-39":2, "40-44":2, 
            "45-49":3, "50-54":3, "55-59":3
        }, inplace=True)
        df.inv_nodes.replace({
            "0-2":0, "3-5":0, "6-8":0, 
            "9-11":1, "12-14":1, "15-17":1, 
            "18-20":2, "21-23":2, "24-26":2, 
            "27-29":3, "30-32":3, "33-35":3, "36-39":3
        }, inplace=True)
        df.node_caps = (df.node_caps == "yes")
        df.breast = (df.breast == "left")
        df.breast_quad.replace({
            "left_up":0, "left_low":1, "right_up":2, 
            "right_low":3, "central":4
        }, inplace=True)
        df.irradiat = (df.irradiat == "yes")
        df = df.astype(int)
        cardinality = [  df[column].max() + 1 for column in df.columns ]
        target = "recurrence"
        monotonicities = [("age", +1), ("menopause", +1), ("deg_malig", +1), ("tumor_size", +1), ("irradiat", -1)]
        return df, cardinality, target, monotonicities
    
    elif name == "diabetes":
        path = os.path.join(base, "diabetes.csv")
        df = pd.read_csv(path)
        for column in df.columns[1:8]:
            df[column].replace(0, np.NaN, inplace=True)
        
        df = df.dropna()
        df.Age = pd.cut(df.Age, [-np.inf, 30, 40, np.inf], labels = np.arange(3))
        df.BMI = pd.cut(df.BMI, [-np.inf, 25, 30, np.inf], labels = np.arange(3))
        df.BloodPressure = pd.cut(df.BloodPressure, [-np.inf, 76.1, 98.1, np.inf], labels = np.arange(3))
        df.DiabetesPedigreeFunction = pd.cut(df.DiabetesPedigreeFunction, [-np.inf, .244, .525, .805, 1.11, np.inf], labels = np.arange(5))
        df.Insulin = pd.cut(df.Insulin, [-np.inf, 75, 150, np.inf], labels = np.arange(3))
        df.Pregnancies = pd.cut(df.Pregnancies, [-np.inf, 2, 6, np.inf], labels = np.arange(3))
        df.Glucose = pd.cut(df.Glucose, [-np.inf, 89, 107, 123, 143, np.inf], labels = np.arange(5))

        df = df.astype(int)
        cardinality = [df[column].max() + 1 for column in df.columns ]
        target = "Outcome"
        monotonicities = [("Age", +1), ("Pregnancies", +1), ("BMI", +1), ("DiabetesPedigreeFunction", +1)]
        return df, cardinality, target, monotonicities
    elif name == "thyroid":
        path = os.path.join(base, "new-thyroid.data")
        columns = ["Hyperthyroid", "T3_resin", "T4", "T3", "TSH", "TSH_diff"]
        df = pd.read_csv(path, names = columns)
        df = df[df.Hyperthyroid.isin((1,2))]
        df.Hyperthyroid = (df.Hyperthyroid == 2)

        for col in ["T3_resin", "T4", "T3", "TSH", "TSH_diff"]:
            df[col] = pd.cut(df[col], 2, labels=np.arange(2))

        target = "Hyperthyroid"
        
        monotonicities = [("T3_resin", +1), ("T3", +1), ("TSH", +1), ("TSH_diff", +1), ("T4", +1)]

        df = df.astype(int)
        cardinality = [ df[column].max() + 1 for column in df.columns ]
        return df, cardinality, target, monotonicities
    elif name == "heart-disease":
        path = os.path.join(base, "processed.cleveland.data")
        columns = ["age", "sex", "cp", "trestbps", "chol", "diabetes", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca",
        "thal", "heart_disease"]
        df = pd.read_csv(path, names=columns, na_values="?").dropna()
        df = df[["sex", "age", "trestbps", "chol", "diabetes", "heart_disease"]]
        df["heart_disease"] = (df.heart_disease.astype(int) != 0).astype(int)
        df.chol = df.chol > 240
        df.trestbps = pd.cut(df.trestbps, [0, 120, 140, np.inf], labels = np.arange(3))
        df.age = pd.cut(df.age, [0, 40, 60, np.inf], labels = np.arange(3))
        target = "heart_disease"
        
        monotonicities = [("sex", +1), ("age", +1), ("trestbps", +1), ("chol",+1), ("diabetes",+1)]

        df = df.astype(int)
        cardinality = [ df[column].max() + 1 for column in df.columns ]
        return df, cardinality, target, monotonicities
    


def get_cis(bn):
    return [
        (list(indep.event1)[0], each, list(indep.event3)[0])
        
        for indep in bn.get_independencies().get_assertions()
        for each in indep.event2
        if len(indep.event3)==1
    ]


def get_bn_dataset(name: str):
    if name in ( "earthquake", "sachs", "asia"):
        bn = BIFReader(f"{name}.bif").get_model()
        train = bn.simulate(n_samples=100, seed=0).astype(int)
        test = bn.simulate(n_samples=100, seed=1).astype(int)
        cis = get_cis(bn)
        r = [bn.get_cardinality(each) for each in train.columns]
        names = train.columns.tolist()
        return r, names, train, test, cis
        


def get_data_loader(df, batch_size, **kwargs):
    dataset = TensorDataset(torch.Tensor(df.to_numpy()))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    return dataset, data_loader

    