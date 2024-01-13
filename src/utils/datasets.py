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
            temp = [i for i in line.strip().split(" ")]
            temp_data = [0] * dim
            count = 0
            for i in temp[:]:
                # ind, val = i.split(':')
                temp_data[count] = float(i)
                count += 1
            data.append(temp_data)
            line = fp.readline()
    X_data = np.array(data, dtype=np.float32)
    return X_data

def gen_dataset(datadir, dset_name, normalize=False, **kwargs):
    if dset_name in ["helix", "helix_short", "helix_uneven", "helix_short_appended", "circle"]:
        trn_file = os.path.join(datadir, dset_name + '.trn')
        val_file = os.path.join(datadir, dset_name + '.val')
        tst_file = os.path.join(datadir, dset_name + '.tst')
        data_dims = 3
        x_trn = csv_file_load(trn_file, dim=data_dims)
        x_val = csv_file_load(val_file, dim=data_dims)
        x_tst = csv_file_load(tst_file, dim=data_dims)

        fullset = CustomDataset(torch.from_numpy(x_trn))
        valset = CustomDataset(torch.from_numpy(x_val))
        testset = CustomDataset(torch.from_numpy(x_tst))

        return fullset, valset, testset

    elif dset_name in ["set-mnist-50","set-mnist-100", "set-fmnist-200"]:
        def shufflerow(tensor, axis):
            row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis)  # get permutation indices
            for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
            row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
            return tensor.gather(axis, row_perm)
        
        fullset = np.load(os.path.join(datadir, 'train_sets.npy'))
        x_trn = fullset
        x_tst = np.load(os.path.join(datadir, 'test_sets.npy'))
        if(normalize):
            fullset = (fullset - fullset.min(axis=1, keepdims=True)) / (fullset.max(axis=1, keepdims=True) - fullset.min(axis=1, keepdims=True))
            x_tst   = (x_tst - x_tst.min(axis=1, keepdims=True)) / (x_tst.max(axis=1, keepdims=True) - x_tst.min(axis=1, keepdims=True))
        fullset = CustomDataset(torch.from_numpy(x_trn))
        valset = CustomDataset(shufflerow(torch.from_numpy(x_trn), axis=1))
        testset = CustomDataset(torch.from_numpy(x_tst))
        print(fullset.data.shape,valset.data.shape,testset.data.shape)
        
        return fullset, valset, testset
