import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, device=None, transform=None):
        self.transform = transform
        if device is not None:
            # Push the entire data to given device, eg: cuda:0
            self.data = data.float().to(device)
        else:
            self.data = data.float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_data = self.data[idx]
        if self.transform is not None:
            sample_data = self.transform(sample_data)
        return sample_data  # .astype('float32')

def csv_file_load(path, dim, save_data=False):
    data = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            temp = [i for i in line.strip().split(",")]
            temp_data = [0] * dim
            count = 0
            for i in temp[:-1]:
                # ind, val = i.split(':')
                temp_data[count] = float(i)
                count += 1
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    return X_data

def gen_dataset(datadir, dset_name, **kwargs):
    if dset_name == "toy_helix":
        np.random.seed(42)
        trn_file = os.path.join(datadir, dset_name + '.trn')
        val_file = os.path.join(datadir, dset_name + '.val')
        tst_file = os.path.join(datadir, dset_name + '.tst')
        data_dims = 2
        x_trn = csv_file_load(trn_file, dim=data_dims)
        x_val = csv_file_load(val_file, dim=data_dims)
        x_tst = csv_file_load(tst_file, dim=data_dims)

        sc = StandardScaler()
        x_trn = sc.fit_transform(x_trn)
        x_val = sc.transform(x_val)
        x_tst = sc.transform(x_tst)

        fullset = CustomDataset(x_trn)
        valset = CustomDataset(x_val)
        testset = CustomDataset(x_tst)

        return fullset, valset, testset
