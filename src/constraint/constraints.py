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


def predict_proba(model, r, data, target_index, marg_indices=None, device='cpu'):
    log_p = torch.zeros((len(data), r[target_index]), device=device)
    log_denom = model(data, (target_index,)) if marg_indices is None else model(data, (target_index,*marg_indices))
    log_denom = log_denom.ravel()
    for i in range(r[target_index]):
        data_i = data.clone()
        data_i[:, target_index] = i
        log_numer = model(data_i) if marg_indices is None else model(data_i, marg_indices)
        log_p[:, i] = log_numer.ravel() - log_denom
    
    return torch.softmax(log_p, axis=1)

class ContextSpecificIndependence(EqualityConstraint):
    def __init__(self, X, Y, Z, z, r):
        # X \indep Y | Z = z
        self.X = X
        self.Y = Y
        self.Z = Z
        self.z = z
        self.r = r
        super().__init__()
    
    def violation(self, model, dataset, config_data, device="cpu", **kwargs):
        # P(X | Y, Z = z) = P(X | Z = z) 
        n_features = len(self.r)
        data = torch.zeros((self.r[self.Y], n_features), device=device)
        for i in range(self.r[self.Y]):
            data[i, self.Y] = i
            data[i, self.Z] = self.z
        
        marg_indices = [i for i in range(n_features) if i not in (self.X, self.Y, self.Z )]
        p1 = predict_proba(model, self.r, data, self.X, marg_indices, device)
        p2 = predict_proba(model, self.r, data, self.X, marg_indices + [self.Y], device)
        delta = self.delta(p1, p2)
        violation = self.degree_violation(delta)
        return violation / (self.r[self.X]*self.r[self.Y])
            
class InequalityConstraint(AbstractConstraint):
    def __init__(self, sign, epsilon):
        super().__init__()
        self.sign = sign
        self.epsilon = epsilon
        
    def delta(self, output_1, output_2):
        delta = torch.sub(output_1, output_2)*self.sign + self.epsilon
        return delta
    def degree_violation(self, delta):
        return torch.sum(torch.max(delta, torch.tensor(0.0, device=delta.device))**2)

class MonotonicityConstraint(InequalityConstraint):
    def __init__(self, Xj, Xi, r, sign, epsilon):
        super().__init__(sign, epsilon)
        self.Xj = Xj
        self.Xi = Xi
        self.r = r
        
    def violation(self, model, dataset, config_data, device="cpu", **kwargs):
        n_features = len(self.r)
        marg_indices = [i for i in range(n_features) if i not in (self.Xi, self.Xj)]
        data = torch.zeros((self.r[self.Xj], n_features), device=device)
        for i in range(self.r[self.Xj]):
            data[i, self.Xj] = i
        cdf = torch.cumsum(predict_proba(model, self.r, data, self.Xi, marg_indices,device=device), axis=1)
        
        total = torch.tensor(0.0, device=device)
        count = 0
        for xi in range(self.r[self.Xi]):
            for xj_ in range(self.r[self.Xj]):
                for xj in range(xj_):
                    # xj_ > xj
                    delta = self.delta(cdf[xj_, xi], cdf[xj, xi])
                    total += self.degree_violation(delta)
                    count += 1
                    
        return torch.div(total, count)

class FalsePositiveConstraint(InequalityConstraint):
    def __init__(self, target, r, epsilon):
        super().__init__(+1, epsilon)
        self.target = target
        self.r = r
        assert self.r[self.target] == 2
    
    def violation(self, model, dataset, config_data, device="cpu", batch_size=64, **kwargs):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )
        total = torch.tensor(0.0, device=device)
        count = 0
        for (data,) in dataloader:
            
            data = data.to(device)
            y = data[:, self.target].clone()
            p = predict_proba(model, self.r, data[y == 1], target_index=self.target, device=device)
            
            p0 = p[:, 0]
            delta = self.delta(p0, 0.5)
            total += self.degree_violation(delta)
            count += 1
            
        return torch.div(total, count)


def log_likelihood(data_loader, model, device="cpu"):
    model.eval()
    total = 0
    for (data,) in data_loader:
        data = data.to(device)
        total += model(data).to("cpu").detach().numpy().sum()
    return total
