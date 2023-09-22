import pandas as pd 
import numpy as np
from enum import Enum
from typing import List 

VariableType = Enum('VariableType', ['Boolean', 'Categorical', 'Continuous'])


class Dataset:
    X: np.array
    
    def __init__(self, X):
        self.X = X 

    def __len__(self):
        return len(X)
    
    def to_dataloader(self):
        pass

class Constraint:
    pass 

class Monotonicity(Constraint):
    C: np.array # an nxn matrix have value 0 for no influence, +1 for positive and -1 for negative influence
    def __init__(self, C):
        self.C = C

class Domain:
    names: List[str]
    types: List[VariableType]
    r: List[float] # (non-continuous) variable i can have values 0...ri
    constraints: List[Constraint]

    def __init__(self, names: List[str], types: List[VariableType],  r: List[float]): 
        self.names = names 
        self.types = types 
        self.r = r 


def fetch_data(name: str) -> Tuple[Domain, Dataset, Dataset]:
    if name == "redwine":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        frame = pd.read_csv(url, sep=";")
        # Fixed
        k = 2
        for name in frame.columns:
        frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
            .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
            .flatten().astype(int)

        X = frame.to_numpy().astype(int)
        X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
        r = [(m + 1) for i, m in enumerate(X.max(axis=0))]
        types = [VariableType.Boolean for _ in r]
        names = frame.columns.tolist()

        domain = Domain(names, types, r)
        train = Dataset(X_train)
        test = Dataset(X_test)

        return domain, train, test

    elif name == "whitewine":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        frame = pd.read_csv(url, sep=";")
        # Fixed
        k = 2
        for name in frame.columns:
        frame[name] = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='kmeans') \
            .fit_transform(frame[name].to_numpy().reshape(-1, 1)) \
            .flatten().astype(int)

        X = frame.to_numpy().astype(int)
        X_train, X_test = train_test_split(X, test_size=0.5, random_state=0, stratify=X[:, -1])
        r = [(m + 1) for i, m in enumerate(X.max(axis=0))]

        types = [VariableType.Boolean for _ in r]
        names = frame.columns.tolist()

        domain = Domain(names, types, r)
        train = Dataset(X_train)
        test = Dataset(X_test)

        return domain, train, test