from abc import ABC, abstractmethod

import torch
from torch import nn
class AbstractConstraint(ABC):
    def __init__(self, device=torch.device("cpu"), batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def violation(self, model: nn.module, data: torch.Tensor, **kwargs) -> float:
        """
        Compute degree of constraint violation for a model over the training data
        :param model: pytorch model
        :param data: training data tensor. shape of [batch, num_features, num_channels]
        :return: A real value representing the degree of constraint violation
        """
        pass

class GeneralizationConstraint(AbstractConstraint):
    def __init__(self, get_sim_data):
        """
        Equality constraint that computes difference in log-likelihood of similar data instances as delta
        :param get_sim_data: A function sim: data \rightarrow (tensor_1, tensor_2) that returns a tuple of tensors that are pairwise similar
        """
        self.get_sim_data = get_sim_data
        super().__init__()

    def violation(self, model: nn.module, data: torch.Tensor, **kwargs):
        sim_data_1, sim_data_2 = self.get_sim_data(data, **kwargs)
        model.eval()
        output_1 = model(sim_data_1)
        output_2 = model(sim_data_2)
        torch.add(output_1, output_2, alpha=-1)
        degree_violation = torch.square(torch.sum(output_1))
        return degree_violation

