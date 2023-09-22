import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from constraint.constraints import GeneralizationConstraint
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
class Train:
    def __init__(self, config_data):
        version = config_data.constraint_args.type
        config_data['version'] = version
        self.cfg = config_data
        logger = SummaryWriter()


    def model_eval_loss(self, data_loader, model, lmbda=0):
        loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(self.cfg.train_args.device), \
                    targets.to(self.cfg.train_args.device, non_blocking=True)
                outputs = model(inputs)

                loss_nll = -outputs.sum() / (inputs.shape[0])
                if lmbda != 0:
                    loss_fn = nn.CrossEntropyLoss(reduction="sum")
                    loss_ce = loss_fn(outputs, targets)

                loss += (1 - lmbda) * loss_nll + lmbda * loss_ce
        loss /= len(data_loader.dataset)

    def make_spn(S, I, R, D, dropout, device) -> RatSpn:
        """Construct the RatSpn"""

        # Setup RatSpnConfig
        config = RatSpnConfig()
        config.F = 28 ** 2
        config.R = R
        config.D = D
        config.I = I
        config.S = S
        config.C = 10
        config.dropout = dropout
        config.leaf_base_class = RatNormal
        config.leaf_base_kwargs = {}

        # Construct RatSpn from config
        model = RatSpn(config)

        model = model.to(device)
        model.train()

        print("Using device:", device)
        return model
