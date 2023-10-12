from abc import ABC, abstractmethod

import torch
from torch import nn


def get_outputs(data_loader, model, device="cpu"):
    outputs = None
    for batch_idx, batch in enumerate(data_loader):
        inputs = batch
        inputs = inputs.to(device)
        if outputs is None:
            outputs = model(inputs)
        else:
            torch.cat((outputs, model(inputs)))
    return outputs

class AbstractConstraint(ABC):
    def __init__(self, device=torch.device("cpu"), batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def violation(self, model, dataset: torch.Tensor, **kwargs) -> float:
        """
        Compute degree of constraint violation for a model over the training data
        :param model: pytorch model
        :param data: training data tensor. shape of [batch, num_features, num_channels]
        :return: A real value representing the degree of constraint violation
        """
        pass

class EqualityConstraint(AbstractConstraint):
    def delta(self, output_1, output_2):
        delta = torch.sub(output_1, output_2)
        return delta
    def degree_violation(self, delta):
        return torch.sum(torch.abs(delta))

class GeneralizationConstraint(EqualityConstraint):
    def __init__(self, get_sim_dataloader):
        """
        Equality constraint that computes difference in log-likelihood of similar data instances as delta
        :param get_sim_dataloader: A function sim: data \rightarrow (dataloader_1, dataloader_2) that returns a tuple
        of dataloaders, elements of which are pairwise similar
        """
        self.get_sim_dataloader = get_sim_dataloader
        super().__init__()

    def violation(self, model, dataset, config_data, device="cpu", **kwargs):
        sim_dataloader_1, sim_dataloader_2 = self.get_sim_dataloader(dataset, config_data, **kwargs)
        output_1 = get_outputs(sim_dataloader_1, model, device=device)
        output_2 = get_outputs(sim_dataloader_2, model, device=device)
        delta = self.delta(output_1, output_2)
        degree_violation = torch.div(self.degree_violation(delta), len(output_1))
        return degree_violation

