import numpy as np
import torch
from .datasets import CustomDataset

def get_sim_dataloader(dataset, config_data, **kwargs):
    cfg = config_data
    mask = None
    if cfg.dataset.name in ['helix', 'helix_short']:
        if 'sim_data_size' in kwargs.keys():
            sim_data_size = kwargs['sim_data_size']
        else:
            sim_data_size = 1000
        x1 = np.linspace(0, 2 * np.pi, sim_data_size)
        y1 = np.sin(x1)
        z1 = np.cos(x1)
        helix = np.stack([x1, y1, z1]).T
        data1 = helix
        sim_dataset_1 = CustomDataset(torch.from_numpy(data1))

        x2 = np.linspace(2 * np.pi, 4 * np.pi, sim_data_size)
        y2 = np.sin(x2)
        z2 = np.cos(x2)
        helix = np.stack([x2, y2, z2]).T
        data2 = helix
        sim_dataset_2 = CustomDataset(torch.from_numpy(data2))
        # dataset_1, dataset_2 = torch.utils.data.random_split(sim_dataset, [int(sim_data_size/2), int(sim_data_size/2)],
        #                                                      torch.Generator().manual_seed(42))
    elif cfg.dataset.name in ['helix_uneven']:
        if 'sim_data_size' in kwargs.keys():
            sim_data_size = kwargs['sim_data_size']
        else:
            sim_data_size = 1000
        x1 = np.linspace(0, 2 * np.pi, sim_data_size)
        y1 = np.sin(x1)
        z1 = np.cos(x1)
        helix = np.stack([x1, y1, z1]).T
        data1 = helix
        sim_dataset_1 = CustomDataset(torch.from_numpy(data1))

        x2 = np.linspace(0 * np.pi, 4 * np.pi, sim_data_size)
        y2 = np.sin(x2)/2
        z2 = np.cos(x2)/2
        helix = np.stack([x2, y2, z2]).T
        data2 = helix
        sim_dataset_2 = CustomDataset(torch.from_numpy(data2))

    elif cfg.dataset.name in ['set-mnist-50', 'set-mnist-100']:
        def shufflerow(tensor, axis):
            row_perm = torch.rand(tensor.shape[:axis+1]).argsort(axis)  # get permutation indices
            for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
            row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
            return tensor.gather(axis, row_perm)
        
        sim_dataset_1 = dataset.repeat(5, 1, 1)
        sim_dataset_2 = shufflerow(sim_dataset_1.clone(), axis=1)
        
        
    dataloader_1, dataloader_2 = torch.utils.data.DataLoader(sim_dataset_1, batch_size=1000,
                                shuffle=False, pin_memory=True), torch.utils.data.DataLoader(sim_dataset_2, batch_size=1000,
                                shuffle=False, pin_memory=True)
    return dataloader_1, dataloader_2


