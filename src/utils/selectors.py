import numpy as np
import torch
def get_sim_dataloader(dataset, config_data, **kwargs):
    cfg = config_data
    mask = None
    if cfg.dataset.name == 'helix':
        if 'atol' in kwargs.keys():
            atol = kwargs['atol']
        else:
            atol = 1e-1
        for x in range(0, 7, 2):
            if mask is None:
                mask = np.isclose(dataset[:, 1], np.sin(x*np.pi), atol=atol)
            else:
                np.append(mask, np.isclose(dataset[:, 1], np.sin(x*np.pi), atol=atol))
        selected_subset = dataset[mask]
        if len(selected_subset) % 2 == 1:
            selected_subset = selected_subset[:-1]
        subsets_size = int(np.floor(len(selected_subset) / 2))
        dataset_1, dataset_2 = torch.utils.data.random_split(selected_subset, [subsets_size, subsets_size],
                                                             torch.Generator().manual_seed(42))
    dataloader_1, dataloader_2 = torch.utils.data.DataLoader(dataset_1, batch_size=1000,
                                shuffle=False, pin_memory=True), torch.utils.data.DataLoader(dataset_2, batch_size=1000,
                                shuffle=False, pin_memory=True)
    return dataloader_1, dataloader_2


