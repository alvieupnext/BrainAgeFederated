from datetime import datetime
from random import random
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

from centralized import (get_test_loader, get_train_valid_loader, group_datasets,
                         train, validate, convert_state_dict, load_model, DEVICE)
from collections import OrderedDict
import flwr as fl
import torch.nn as nn
import pandas as pd
import torch

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# net = load_model()
#
# trainloader, valloader = get_train_valid_loader('patients_dataset_6326_train.csv', batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1)


project_name = 'test'
save_dir = './utils/models/' + project_name + "/"

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net, trainloader, valloader, cid, name=None):
      self.net = net
      self.trainloader = trainloader
      self.valloader = valloader
      self.cid = cid
      self.name = name

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        friendly_name = str(self.name) or str(self.cid)
        client_save_dir = save_dir + friendly_name + "/"
        #If the repository does not exist, create it
        if not os.path.exists(client_save_dir):
            os.makedirs(client_save_dir)
        model_save_path = client_save_dir + datetime.now().strftime('{}_%d-%m-%y-%H_%M.pt'.format(project_name))
        train(self.net, self.trainloader, self.valloader, 1, model_save_path)
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        val_losses, _, _, _, _, val_mae = validate(self.net, self.valloader)
        return float(val_losses), len(self.valloader), {}