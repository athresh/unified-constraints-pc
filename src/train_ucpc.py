import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from utils.datasets import gen_dataset
from utils.config_utils import load_config_data
import time
import argparse

class Train:
    def __init__(self, config_data):
        version = config_data.constraint_args.type
        config_data['version'] = version
        self.cfg = config_data
        self.logger = SummaryWriter()


    def model_eval_loss(self, data_loader, model, lmbda=0):
        loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs = batch
                inputs = inputs.to(self.cfg.train_args.device)
                outputs = model(inputs)
                loss = -outputs.sum() / (inputs.shape[0])

                # loss_nll = -outputs.sum() / (inputs.shape[0])
                # if lmbda != 0:
                #     loss_fn = nn.CrossEntropyLoss(reduction="sum")
                #     loss_ce = loss_fn(outputs, targets)
                #
                # loss += (1 - lmbda) * loss_nll + lmbda * loss_ce
        loss /= len(data_loader.dataset)
        return loss

    def make_spn(S, I, R, D, F, device) -> RatSpn:
        """Construct the RatSpn"""

        # Setup RatSpnConfig
        config = RatSpnConfig()
        config.F = F
        config.R = R
        config.D = D
        config.I = I
        config.S = S
        config.C = 10
        config.leaf_base_class = RatNormal
        config.leaf_base_kwargs = {}

        # Construct RatSpn from config
        model = RatSpn(config)

        model = model.to(device)
        model.train()

        print("Using device:", device)
        return model

    def train(self):
        """
        General training loop
        """
        logger = self.logger
        trainset, validset, testset = gen_dataset(self.cfg.dataset.datadir,
                                                           self.cfg.dataset.name)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=True, pin_memory=True)

        tstloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        model = self.make_spn(self.cfg.model.S, self.cfg.model.I, self.cfg.model.R, self.cfg.model.D, self.cfg.model.F, self.cfg.train_args.device)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.train_args.lr)

        for epoch in range(self.cfg.train_args.num_epochs):
            model.train()
            start_time = time.time()

            for batch_idx, batch in enumerate(trainloader):
                inputs = batch.to(self.cfg.train_args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = -outputs.sum() / (inputs.shape[0])
                loss.backward()
                optimizer.step()
            epoch_time = time.time() - start_time
            print_args = self.cfg.train_args.print_args

        """
        ################################################# Evaluation Loop #################################################
        """
        trn_losses = []
        val_losses = []
        tst_losses = []
        if (epoch + 1) % self.cfg.train_args.print_every == 0:
            trn_loss = 0
            val_loss = 0
            tst_loss = 0
            model.eval()
            if ("trn_loss" in print_args):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(trainloader):
                        inputs = batch.to(self.cfg.train_args.device)
                        outputs = model(inputs)
                        loss = -outputs.sum() / (inputs.shape[0])
                        trn_loss += loss.item()
                    trn_losses.append(trn_loss)
            if ("val_loss" in print_args):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(valloader):
                        inputs = batch.to(self.cfg.train_args.device)
                        outputs = model(inputs)
                        loss = -outputs.sum() / (inputs.shape[0])
                        val_loss += loss.item()
                    val_losses.append(trn_loss)
            if ("tst_loss" in print_args):
                with torch.no_grad():
                    for batch_idx, batch in enumerate(tstloader):
                        inputs = batch.to(self.cfg.train_args.device)
                        outputs = model(inputs)
                        loss = -outputs.sum() / (inputs.shape[0])
                        tst_loss += loss.item()
                    tst_losses.append(trn_loss)
        if ("trn_loss" in print_args):
            self.logger.add_scalar('Train loss', trn_losses, epoch)
        if ("val_loss" in print_args):
            self.logger.add_scalar('Val loss', val_losses, epoch)
        if ("tst_loss" in print_args):
            self.logger.add_scalar('Test loss', val_losses, epoch)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file
    config_data = load_config_data(args.config_file)
    model = Train(config_data)
    model.train()