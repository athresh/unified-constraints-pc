import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from packages.spn.experiments.RandomSPNs_layerwise.rat_spn import RatSpn, RatSpnConfig
from packages.spn.experiments.RandomSPNs_layerwise.distributions import RatNormal
from packages.spn.algorithms.layerwise.distributions import *
from utils.datasets import gen_dataset
from utils.config_utils import load_config_data
from utils.utils import visualize_3d, visualize_set_image
from utils.selectors import get_sim_dataloader
from constraint.constraints import GeneralizationConstraint
import time
import argparse
from pathlib import Path
import os
from packages.pfc.models import EinsumNet, LinearSplineEinsumFlow
from packages.pfc.config import EinetConfig
from packages.pfc.components.spn.Graph import random_binary_trees, poon_domingos_structure
import tqdm 
import pickle
import random
import sys 

class Train:
    def __init__(self, config_data):
        version = config_data.constraint_args.type
        config_data['version'] = version
        self.cfg = config_data
        # self.log_dir = "runs/{}/{}".format(self.cfg.dataset.name, self.cfg.model.name)
        # p = Path(self.log_dir)
        # p.mkdir(parents=True, exist_ok=True)
        log_comment = "_{}_{}".format(self.cfg.dataset.name, self.cfg.model.name)
        self.logger = SummaryWriter(comment=log_comment) if not hasattr(config_data, 'experiment_dir') else SummaryWriter(os.path.join(config_data['experiment_dir'],'results'))

    def model_eval_loss(self, data_loader, model, lmbda=0):
        loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inputs = batch
                inputs = inputs.to(self.cfg.train_args.device)
                outputs = model(inputs)
                loss += outputs.sum()
        loss /= len(data_loader.dataset)
        return loss.item()

    
    def make_pfc(self, model_name, num_sums, num_input_distributions, num_repetition, depth, num_vars, num_dims, num_classes, graph_type, leaf_type, leaf_config, device)-> EinsumNet:
        config = EinetConfig()
        config.num_input_distributions = num_input_distributions
        config.num_repetition = num_repetition
        config.num_sums = num_sums
        config.graph_type  = graph_type
        config.depth = depth
        config.num_classes = num_classes
        config.device = device
        config.num_vars = num_vars
        config.num_dims = num_dims
        config.leaf_type = leaf_type
        config.leaf_config = leaf_config
        config.graph = random_binary_trees(config.num_vars, config.depth, config.num_repetition) if graph_type == "random_binary_trees" else None
        model = LinearSplineEinsumFlow(config).to(device) if "Flow" in model_name else EinsumNet(config).to(device)
        return model 
    
    def make_spn(self, S, I, R, D, F, C, leaf_type, leaf_config, device) -> RatSpn:
        """Construct the RatSpn"""

        # Setup RatSpnConfig
        config = RatSpnConfig()
        config.F = F
        config.R = R
        config.D = D
        config.I = I
        config.S = S
        config.C = C
        config.dropout = 0.0
        config.leaf_base_class = leaf_type
        config.leaf_base_kwargs = leaf_config

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
        normalize = self.cfg.dataset.normalize if hasattr(self.cfg.dataset, 'normalize') else False
        trainset, validset, testset = gen_dataset(
                            self.cfg.dataset.datadir,
                            self.cfg.dataset.name,
                            normalize=normalize)

        trn_batch_size = self.cfg.dataloader.batch_size
        val_batch_size = self.cfg.dataloader.batch_size
        tst_batch_size = self.cfg.dataloader.batch_size


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trn_batch_size,
                                                  shuffle=False, pin_memory=True)

        valloader = torch.utils.data.DataLoader(validset, batch_size=val_batch_size,
                                                shuffle=True, pin_memory=True)

        tstloader = torch.utils.data.DataLoader(testset, batch_size=tst_batch_size,
                                                 shuffle=False, pin_memory=True)

        if self.cfg.model.name in ["RatSPN", "RatSPN_constrained"]:
            model = self.make_spn(self.cfg.model.S, self.cfg.model.I,
                                  self.cfg.model.R, self.cfg.model.D,
                                  self.cfg.model.F, self.cfg.model.C,
                                  eval(self.cfg.model.leaf_type),
                                  self.cfg.model.leaf_config,
                                  self.cfg.train_args.device)
        elif self.cfg.model.name in ["EinsumNet","EinsumFlow"]:
            model = self.make_pfc(
                        self.cfg.model.name,
                        self.cfg.model.num_sums,
                        self.cfg.model.num_input_distributions,
                        self.cfg.model.num_repetition,
                        self.cfg.model.depth,
                        self.cfg.model.num_vars,
                        self.cfg.model.num_dims,
                        self.cfg.model.num_classes,
                        self.cfg.model.graph_type,
                        self.cfg.model.leaf_type,
                        self.cfg.model.leaf_config,
                        self.cfg.train_args.device
                    )
        else:
            raise ValueError("Model not defined")
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.train_args.lr)
        trn_losses = []
        val_losses = []
        tst_losses = []
        epochs = []
        lmbda = self.cfg.constraint_args.lmbda
        for epoch in range(self.cfg.train_args.num_epochs+1):
            model.train()
            start_time = time.time()
            total_violation = 0
            for batch_idx, batch in enumerate(trainloader):
                inputs = batch.to(self.cfg.train_args.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                if self.cfg.constraint_args.constrained:
                    # if (epoch + 1) % 10 == 0:
                    #     lmbda = np.min([lmbda * 2, 100000])
                    if self.cfg.constraint_args.type == "generalization":
                        constraint = GeneralizationConstraint(get_sim_dataloader)
                    else:
                        raise ValueError("Constraint not defined")
                    violation = constraint.violation(model, batch, config_data=self.cfg, device=self.cfg.train_args.device, **self.cfg.constraint_args)
                    total_violation += violation.item()
                    loss = torch.add(torch.div(-outputs.sum(), inputs.shape[0]), (torch.mul(violation, lmbda)))
                    # if (epoch + 1) % 10 == 0:
                    #     print(-outputs.sum())
                    #     print(violation)
                    #     print(loss)
                    #     print(lmbda)
                    #     print(inputs.shape[0])
                else:
                    loss = -outputs.sum() # / inputs.shape[0]
                loss.backward()
                optimizer.step()
            epoch_time = time.time() - start_time
            print_args = self.cfg.train_args.print_args
            total_violation /= self.cfg.train_args.num_epochs
            """
            ################################################# Evaluation Loop #################################################
            """
            if (epoch) % self.cfg.train_args.print_every == 0:
                trn_loss = 0
                val_loss = 0
                tst_loss = 0
                epochs += [epoch]
                model.eval()
                if ("trn_loss" in print_args):
                    trn_loss = self.model_eval_loss(trainloader, model)
                    trn_losses.append(trn_loss)
                    self.logger.add_scalar('Train loss', np.array(trn_loss), epoch)
                if ("val_loss" in print_args):
                    val_loss = self.model_eval_loss(valloader, model)
                    val_losses.append(val_loss)
                    self.logger.add_scalar('Val loss', np.array(val_loss), epoch)
                if ("tst_loss" in print_args):
                    tst_loss = self.model_eval_loss(tstloader, model)
                    tst_losses.append(tst_loss)
                    self.logger.add_scalar('Test loss', np.array(tst_loss), epoch)
                if self.cfg.constraint_args.constrained:
                    self.logger.add_scalar('Violation', total_violation, epoch)
                print("Epoch: {} | Train loss: {:.4f} | Val loss: {:.4f} | Test loss: {:.4f} | Violation: {:.4f} | Time: {:.4f}".format(epoch, trn_loss, val_loss, tst_loss, total_violation, epoch_time))
                sys.stdout.flush()
                
                # Save all losses as dict 
                # pickle.dump({'trn_loss': trn_losses, 'val_loss': val_losses, 'tst_loss': tst_losses, 'epochs':epochs}, open(self.cfg.train_args.results_dir+'/losses.pkl', 'wb'))
                 
            if self.cfg.train_args.visualize:
                if (epoch) % self.cfg.train_args.visualize_every == 0:
                    p = Path(self.cfg.train_args.plots_dir)
                    p.mkdir(parents=True, exist_ok=True)
                    if(self.cfg.dataset.name in ["set-mnist-50","set-mnist-100","set-fmnist-200"]):
                        visualize_set_image(model, dataset=trainset,save_dir=self.cfg.train_args.plots_dir, epoch=epoch)
                    elif(self.cfg.dataset.name in ["helix", "helix_short", "helix_uneven","helix_short_appended", "circle"]):
                        visualize_3d(model, dataset=trainset,save_dir=self.cfg.train_args.plots_dir, epoch=epoch)
        self.logger.close()
        p = Path(self.cfg.train_args.save_model_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.cfg.train_args.save_model_dir+'/model.mdl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--trial', type=int)
    parser.add_argument('--sim_data_size', type=int, default=100)
    args = parser.parse_args()
    config_file = args.config_file
    config_data = load_config_data(args.config_file)
    config_data.constraint_args['sim_data_size'] = args.sim_data_size
    
    random.seed(config_data.seed)
    np.random.seed(config_data.seed)
    torch.manual_seed(config_data.seed)


    model = Train(config_data)
    model.train()
