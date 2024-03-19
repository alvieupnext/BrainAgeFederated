from datetime import datetime
from random import random
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from centralized import (get_test_loader, get_train_valid_loader, group_datasets,
                         train, validate, convert_state_dict, load_model, DEVICE, check_improvement, update_loss_df)
from collections import OrderedDict
import flwr as fl
import torch.nn as nn
import pandas as pd
import torch

from plot import plot_age_distribution


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# net = load_model()
#
# trainloader, valloader = get_train_valid_loader('patients_dataset_6326_train.csv', batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1)


# project_name = 'test'
# save_dir = './utils/models/' + project_name + "/"

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net, project_name, save_dir, dataset, cid, name=None):
      self.net = net
      self.dataset = dataset
      #Split the dataset into train and validation
      trainloader, valloader = get_train_valid_loader(self.dataset, batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1)
      self.trainloader = trainloader
      self.valloader = valloader
      self.cid = cid
      self.name = name
      self.project_name = project_name
      self.save_dir = save_dir
      #Plot the data distribution of the dataset
      self.plot_age_distribution()

    def plot_age_distribution(self):
      friendly_name = str(self.name) or str(self.cid)
      client_save_dir = self.save_dir + friendly_name + "/"
      #Create a save dir to a file called age_distribution.pdf
      if not os.path.exists(client_save_dir):
          os.makedirs(client_save_dir)
      plot_save_dir = client_save_dir + "age_distribution.pdf"
      plot_age_distribution(self.dataset, plot_save_dir)

    def get_parameters(self, config):
        print(f"[Client {self.cid}, friendly name {self.name}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        set_parameters(self.net, parameters)

      #Improvement: only keep the latest model that saw improvement on the validation set
    def train(self, config, model_save_path, losses_save_path, criterion, optimizer, scheduler, parameters, patience=5):
        best_loss = 1e9
        num_bad_epochs = 0
        is_new_best = False
        server_round = config["server_round"]
        epochs = config["local_epochs"]
        #Set the parameters of the model to the parameters received
        self.set_parameters(parameters)

        for epoch in range(epochs):
          if num_bad_epochs >= patience:
            print(f"Model reached patience: {patience}")
            # break
          train_loss = self.train_epoch(criterion, optimizer)
          val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(self.net, self.valloader)
          update_loss_df(losses_save_path, server_round, epoch, train_loss, val_loss)
          is_new_best, best_loss, num_bad_epochs = check_improvement(val_loss, best_loss, self.net, model_save_path, epoch,
                                                                     num_bad_epochs)
          scheduler.step(val_loss)
          lr = optimizer.param_groups[0]['lr']
          print(
            f'Epoch: {epoch + 1} of {epochs}, lr: {lr:.2E}, train loss: {train_loss:.2f}, valid loss: {val_loss:.2f}, corr: {corr:.2f}, best loss {best_loss:.2f}, number of epochs without improvement: {num_bad_epochs}')

        return is_new_best, model_save_path

    def train_epoch(self, criterion, optimizer):
      self.net.train()
      train_loss = 0.0
      train_count = 0
      for data in tqdm(self.trainloader, leave=False):
        im, age, _ = data
        im = im.to(device=DEVICE, dtype=torch.float)
        age = age.to(device=DEVICE, dtype=torch.float).reshape(-1, 1)
        optimizer.zero_grad()
        pred_age = self.net(im)
        loss = criterion(pred_age, age)
        loss.backward()
        optimizer.step()
        train_count += im.shape[0]
        train_loss += loss.sum().detach().item()
      train_loss /= train_count
      return train_loss

    def fit(self, parameters, config):
      # Read values from config
        server_round = config["server_round"]
        patience = config["patience"]

        print(f"[Client {self.cid}, friendly name {self.name}, round {server_round}] fit, config: {config}")
        friendly_name = str(self.name) or str(self.cid)
        client_save_dir = self.save_dir + friendly_name + "/"
        #If the repository does not exist, create it
        if not os.path.exists(client_save_dir):
            os.makedirs(client_save_dir)
        model_save_path = client_save_dir + f"{self.name}_{server_round}.pt"
        #Save the losses in a file
        losses_save_path = client_save_dir + str(self.name) + '_losses.csv'
        #If the losses file does not exist, create it with the headers
        # Create a new dataframe with the following columns: server_round, epoch, train_loss, val_loss, train_mae, val_mae
        if not os.path.exists(losses_save_path):
            with open(losses_save_path, 'w') as f:
                f.write('server_round,epoch,train_loss,val_loss,time\n')

        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
      # #Make a copy of the global parameters
      #   global_params = [param.clone().cpu().detach() for param in self.net.parameters()]
      #Get the save path and is new best from train
        is_new_best, save_path = self.train(config, model_save_path, losses_save_path, criterion, optimizer, scheduler, parameters, patience=patience)
      #If the model is not the new best, reload the model from the last best
        if not is_new_best:
            print("Loading the last best model")
            self.net.load_state_dict(torch.load(save_path))
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
      # return 0.1, 1, {}
        set_parameters(self.net, parameters)
        val_losses, _, _, _, _, val_mae = validate(self.net, self.valloader)
        return float(val_losses), len(self.valloader), {}

#Client class for FedProx
class FedProxClient(FlowerClient):
  #Copy the train method from the parent class
  def train(self, config, model_save_path, losses_save_path, criterion, optimizer, scheduler, parameters, patience=5):
    # Read values from config
    server_round = config["server_round"]
    epochs = config["local_epochs"]
    proximal_mu = config["proximal_mu"]
    # Set the global parameters to the parameters received
    self.set_parameters(parameters)
    global_params = [val.detach().clone() for val in self.net.parameters()]

    best_loss = 1e9
    num_bad_epochs = 0
    is_new_best = False

    ##In FedProx, we do not need to override the net parameters, as we are using the proximal term

    for epoch in range(epochs):
      if num_bad_epochs >= patience:
        print(f"Model reached patience: {patience}")
        # break
      train_loss = self.train_epoch_proximal(criterion, optimizer, proximal_mu, global_params)
      val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(self.net, self.valloader)
      update_loss_df(losses_save_path, server_round, epoch, train_loss, val_loss)
      is_new_best, best_loss, num_bad_epochs = check_improvement(val_loss, best_loss, self.net, model_save_path, epoch,
                                                                 num_bad_epochs)
      scheduler.step(val_loss)
      lr = optimizer.param_groups[0]['lr']
      print(
        f'Epoch: {epoch + 1} of {epochs}, lr: {lr:.2E}, train loss: {train_loss:.2f}, valid loss: {val_loss:.2f}, corr: {corr:.2f}, best loss {best_loss:.2f}, number of epochs without improvement: {num_bad_epochs}')

    return is_new_best, model_save_path

  def train_epoch_proximal(self, criterion, optimizer, proximal_mu, parameters):
    self.net.train()
    train_loss = 0.0
    train_count = 0
    for data in tqdm(self.trainloader, leave=False):
      im, age, _ = data
      im = im.to(device=DEVICE, dtype=torch.float)
      age = age.to(device=DEVICE, dtype=torch.float).reshape(-1, 1)
      optimizer.zero_grad()
      proximal_term = 0.0
      for local_weights, global_weights in zip(self.net.parameters(), parameters):
        proximal_term += torch.square((local_weights - global_weights).norm(2))
      pred_age = self.net(im)
      loss = criterion(pred_age, age) + (proximal_mu / 2) * proximal_term
      loss.backward()
      optimizer.step()
      train_count += im.shape[0]
      train_loss += loss.sum().detach().item()
    train_loss /= train_count
    return train_loss