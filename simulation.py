from flwr.server.client_proxy import ClientProxy

from centralized import load_model, DEVICE, get_test_loader, group_datasets, get_train_valid_loader, validate
from client import FlowerClient, set_parameters
import pandas as pd
import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes, Parameters
from typing import Dict, Optional, Tuple, List, Union, OrderedDict
import os
import numpy as np
import torch

from strategy import SaveFedAvg, SaveFedProx
from utils import project_name, save_dir, dwood

# Load patients_dataset_6326_train.csv
df = pd.read_csv('patients_dataset_6326_train.csv')
#Group the dataframe in different dataframes by dataset attribute
dfs = group_datasets(df, mode='dataset')
# #Remove PDD from the dictionary
# dfs.pop('PDD')
dataloaders = {name: get_train_valid_loader(df, batch_size=3, random_seed=10, dataset_scale=1) for name, df in dfs.items()}
names = list(dataloaders.keys())
print(names)
print(dataloaders)
# #Print the dataframe from Other and from PDD
# print(dfs.get('PDD'))
print("Loaded test data")
testdf = pd.read_csv('patients_dataset_6326_test.csv')
testloader = get_test_loader(testdf, batch_size=4, dataset_scale=1)
def client_fn(cid: str) -> FlowerClient:
  """Create a Flower client representing a single organization."""

  # Load model
  net = load_model().to(DEVICE)

  # Dataloaders is a dict with name as key and a tuple with trainloader and valloader as value
  name = names[int(cid)]
  trainloader, valloader = dataloaders[name]


  # Create a  single Flower client representing a single organization
  return FlowerClient(net, project_name, trainloader, valloader, cid, name)

#Evaluation server side using test csv
def get_evaluate_fn(model):
  def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    print("Evaluating round", server_round)
    set_parameters(model, parameters)
    val_losses, _, _, _, _, val_mae = validate(model, testloader)
    # If the repository does not exist, create it
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    #Write the losses to a file in save_dir
    with open(save_dir + 'centralized_losses.txt', 'a') as f:
      f.write(f"{server_round},{val_losses}\n")

    return float(val_losses), {}

  return evaluate

def fit_config(server_round: int):
  """Return training configuration dict for each round.

  Perform two rounds of training with one local epoch, increase to two local
  epochs afterwards.
  """
  config = {
    "server_round": server_round,  # The current round of federated learning
    "local_epochs": 1 if server_round < 2 else 2,  #
  }
  return config

# Specify the resources each of your clients need. By default, each
# client will be allocated 1x CPU and 0x GPUs
client_resources = {"num_cpus": 1, "num_gpus": 0.0}
print(DEVICE.type)
if DEVICE.type == "cuda":
    # here we are asigning an entire GPU for each client.
    client_resources = {"num_cpus": 1, "num_gpus": 1.0}
    # Refer to our documentation for more details about Flower Simulations
    # and how to setup these `client_resources`.
dwood_seed_2 = dwood + 'seed_67.pt'
net = load_model(dwood_seed_2).to(DEVICE)

weights = [val.cpu().numpy() for _, val in net.state_dict().items()]

parameters = fl.common.ndarrays_to_parameters(weights)


# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    evaluate_fn=get_evaluate_fn(net),
    on_fit_config_fn=fit_config,  # Pass the fit_config function
    initial_parameters=parameters,
    # min_fit_clients=10,  # Never sample less than 10 clients for training
    # min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
    # min_available_clients=10,  # Wait until all 10 clients are available
)

fedavg = SaveFedAvg(
  fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    evaluate_fn=get_evaluate_fn(net),
    on_fit_config_fn=fit_config,
  initial_parameters=parameters,
)

fedprox = SaveFedProx(
  fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    evaluate_fn=get_evaluate_fn(net),
    on_fit_config_fn=fit_config,
  initial_parameters=parameters,
  proximal_mu=1.0,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(dfs),
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fedavg,
    client_resources=client_resources,
)