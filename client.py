from datetime import datetime
from random import random
import os

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from centralized import (get_test_loader, get_train_valid_loader, group_datasets,
                         train, validate, convert_state_dict, load_model, check_improvement, update_loss_df)
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

    def __init__(self, net, project_name, save_dir, dataset, cid, name=None, kcrossval=10, device='cuda'):
      self.net = net
      self.dataset = dataset
      #Split the dataset into train and validation
      trainloaders, valloaders = get_train_valid_loader(self.dataset, batch_size=4, random_seed=10, aug='none', kcrossval=kcrossval, icross=-1)
      self.trainloaders = trainloaders
      self.valloaders = valloaders
      self.folds = kcrossval
      self.cid = cid
      self.name = name
      self.project_name = project_name
      self.save_dir = save_dir
      self.device = device
      #Plot the data distribution of the dataset
      self.plot_age_distribution()
      #Save the patient ids in a text file
      self.save_patient_ids()

    def save_patient_ids(self):
      friendly_name = str(self.name) or str(self.cid)
      client_save_dir = self.save_dir + friendly_name + "/"
      if not os.path.exists(client_save_dir):
          os.makedirs(client_save_dir)
      ids_save_dir = client_save_dir + "patient_ids.txt"
      ids = self.dataset['ID']
      with open(ids_save_dir, 'w') as f:
          for item in ids:
              f.write("%s\n" % item)

    def plot_age_distribution(self):
      friendly_name = str(self.name) or str(self.cid)
      client_save_dir = self.save_dir + friendly_name + "/"
      #Create a save dir to a file called age_distribution.pdf
      if not os.path.exists(client_save_dir):
          os.makedirs(client_save_dir)
      plot_save_dir = client_save_dir + "age_distribution.pdf"
      plot_age_distribution(self.dataset, plot_save_dir, friendly_name)

    def get_parameters(self, config):
        print(f"[Client {self.cid}, friendly name {self.name}] get_parameters")
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        set_parameters(self.net, parameters)

      #Improvement: only keep the latest model that saw improvement on the validation set
    def train(self, config, trainloader, valloader, model_save_path, losses_save_path, criterion, optimizer, scheduler, parameters, patience=5):
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
          train_loss = self.train_epoch(trainloader, criterion, optimizer)
          val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(self.net, valloader, self.device)
          update_loss_df(losses_save_path, server_round, epoch, train_loss, val_loss)
          is_new_best, best_loss, num_bad_epochs = check_improvement(val_loss, best_loss, self.net, model_save_path, epoch,
                                                                     num_bad_epochs)
          scheduler.step(val_loss)
          lr = optimizer.param_groups[0]['lr']
          print(
            f'Epoch: {epoch + 1} of {epochs}, lr: {lr:.2E}, train loss: {train_loss:.2f}, valid loss: {val_loss:.2f}, corr: {corr:.2f}, best loss {best_loss:.2f}, number of epochs without improvement: {num_bad_epochs}')

        return is_new_best, model_save_path, best_loss

    def train_epoch(self, trainloader, criterion, optimizer):
      self.net.train()
      train_loss = 0.0
      train_count = 0
      for data in tqdm(trainloader, leave=False):
        im, age, _ = data
        im = im.to(device=self.device, dtype=torch.float)
        age = age.to(device=self.device, dtype=torch.float).reshape(-1, 1)
        optimizer.zero_grad()
        pred_age = self.net(im)
        loss = criterion(pred_age, age)
        loss.backward()
        optimizer.step()
        train_count += im.shape[0]
        train_loss += loss.sum().detach().item()
      #Average the loss, make sure to not divide by zero
      train_loss /= train_count
      return train_loss

    #Function that prepares the model for training
    def initialize_path(self, client_save_dir, server_round, fold=None, central=False):
      # If a fold is defined, store everything in a subfolder
      if fold is not None:
        client_save_dir += f"fold_{fold}/"
        if not os.path.exists(client_save_dir):
          os.makedirs(client_save_dir)
      pt_name = f"{self.name}_{server_round}"
      #If a fold is defined, add the fold to the name
      if fold is not None:
        pt_name += f"_fold_{fold}"
      model_save_path = client_save_dir + f"{pt_name}.pt"
      # Save the losses in a file
      losses_name = client_save_dir + str(self.name) + '_losses'
      if fold is not None:
        losses_name += f'_fold_{fold}'
      suffix = '.txt' if central else '.csv'
      losses_save_path = losses_name + suffix
      # If the losses file does not exist, create it with the headers
      # Create a new dataframe with the following columns: server_round, epoch, train_loss, val_loss, train_mae, val_mae
      #Only do this if your model is not a master model
      if not central and not os.path.exists(losses_save_path):
        with open(losses_save_path, 'w') as f:
          f.write('server_round,epoch,train_loss,val_loss,time\n')
      return model_save_path, losses_save_path


    def fit(self, parameters, config):
      # Read values from config
        server_round = config["server_round"]
        patience = config["patience"]

        print(f"[Client {self.cid}, friendly name {self.name}, round {server_round}, folds {self.folds}] fit, config: {config}")
        friendly_name = str(self.name) or str(self.cid)
        client_save_dir = self.save_dir + friendly_name + "/"
      # If the repository does not exist, create it
        if not os.path.exists(client_save_dir):
          os.makedirs(client_save_dir)
        model_save_path, losses_save_path = self.initialize_path(client_save_dir, server_round, central=True)
        model_save_paths = []
        best_losses = []
        for k in range(self.folds):
          #Initialize the path for the model and the losses
          fold_model_save_path, fold_losses_save_path = self.initialize_path(client_save_dir, server_round, k)
          criterion = nn.L1Loss()
          optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
          scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
          trainloader = self.trainloaders[k]
          valloader = self.valloaders[k]
      # #Make a copy of the global parameters
      #   global_params = [param.clone().cpu().detach() for param in self.net.parameters()]
      #Get the save path and is new best from train
          is_new_best, save_path, best_loss = self.train(config, trainloader, valloader, fold_model_save_path, fold_losses_save_path, criterion, optimizer, scheduler, parameters, patience=patience)
          model_save_paths.append(save_path)
          best_losses.append(best_loss)
        #At the end of the training, reset the parameters to the global parameters
          # set_parameters(self.net, global_params)
      #If the model is not the new best, reload the model from the last best
          # if not is_new_best:
          #     print("Loading the last best model")
          #     self.net.load_state_dict(torch.load(save_path))
      # From the model_save_paths, load every model and average the parameters
        parameters = []
        for path in model_save_paths:
          self.net.load_state_dict(torch.load(path))
          parameters.append([val.cpu().numpy() for _, val in self.net.state_dict().items()])
        avg_parameters = [sum(x) / len(x) for x in zip(*parameters)]
        #Load the model with the average parameters
        set_parameters(self.net, avg_parameters)
      #Save the model with the average parameters
        torch.save(self.net.state_dict(), model_save_path)
      #Validate the model
        avg_val_loss, avg_val_length, val_losses = self.validate()
      #Turn the val losses into a numpy array
        val_losses = np.array(val_losses)
      #From val_losses, obtain the standard deviation
        val_std = np.std(val_losses)
      #To the loss file, add the server round followed by the average validation loss
        with open(losses_save_path, 'a') as f:
          f.write(f"{server_round},{avg_val_loss},{val_std}\n")
        return avg_parameters, len(self.dataset), {}

    def evaluate(self, parameters, config):
      # return 0.1, 1, {}
        set_parameters(self.net, parameters)
        avg_val_loss, avg_val_length, _ = self.validate()
        return float(avg_val_loss), int(avg_val_length), {}

    #Use cross-validation to evaluate the model
    def validate(self):
      val_losses = []
      val_lengths = []
      for valloader in self.valloaders:
        val_loss, _, _, _, _, val_mae = validate(self.net, valloader, self.device)
        val_losses.append(val_loss)
        val_lengths.append(len(valloader))
      # val_losses, _, _, _, _, val_mae = validate(self.net, self.valloader)
      # Average the validation losses
      avg_val_loss = sum(val_losses) / len(val_lengths)
      avg_val_length = sum(val_lengths) / len(val_lengths)
      return float(avg_val_loss), avg_val_length, val_losses

#Client class for FedProx
class FedProxClient(FlowerClient):
  #Copy the train method from the parent class
  def train(self, config, trainloader, valloader,model_save_path, losses_save_path, criterion, optimizer, scheduler, parameters, patience=5):
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
      train_loss = self.train_epoch_proximal(criterion, optimizer, trainloader, proximal_mu, global_params)
      val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(self.net, valloader, self.device)
      update_loss_df(losses_save_path, server_round, epoch, train_loss, val_loss)
      is_new_best, best_loss, num_bad_epochs = check_improvement(val_loss, best_loss, self.net, model_save_path, epoch,
                                                                 num_bad_epochs)
      scheduler.step(val_loss)
      lr = optimizer.param_groups[0]['lr']
      print(
        f'Epoch: {epoch + 1} of {epochs}, lr: {lr:.2E}, train loss: {train_loss:.2f}, valid loss: {val_loss:.2f}, corr: {corr:.2f}, best loss {best_loss:.2f}, number of epochs without improvement: {num_bad_epochs}')

    return is_new_best, model_save_path, best_loss

  def train_epoch_proximal(self, criterion, optimizer, trainloader, proximal_mu, parameters):
    self.net.train()
    train_loss = 0.0
    train_count = 0
    for data in tqdm(trainloader, leave=False):
      im, age, _ = data
      im = im.to(device=self.device, dtype=torch.float)
      age = age.to(device=self.device, dtype=torch.float).reshape(-1, 1)
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