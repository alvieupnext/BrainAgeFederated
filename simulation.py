import argparse

from flwr.server.client_proxy import ClientProxy

from centralized import load_model, DEVICE, get_test_loader, group_datasets, get_train_valid_loader, validate
from client import FlowerClient, set_parameters, FedProxClient
import pandas as pd
import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes, Parameters
from typing import Dict, Optional, Tuple, List, Union, OrderedDict
import os
import numpy as np
import torch

from strategy import SaveFedAvg, SaveFedProx
from utils import dwood

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

#Generate a client function which takes the project name and returns a function that creates a FlowerClient
def gen_client_fn(project_name, strategy, save_dir):
  def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = load_model().to(DEVICE)

    # Dataloaders is a dict with name as key and a tuple with trainloader and valloader as value
    name = names[int(cid)]
    trainloader, valloader = dataloaders[name]

    if strategy == 'FedAvg':
      # Create a  single Flower client representing a single organization
      return FlowerClient(net, project_name, save_dir, trainloader, valloader, cid, name)
    elif strategy == 'FedProx':
      # Create a  single FedProx representing a single organization
      return FedProxClient(net, project_name, save_dir, trainloader, valloader, cid, name)
  return client_fn

#Evaluation server side using test csv
def get_evaluate_fn(model, save_dir):
  def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    print("Evaluating round", server_round)
    set_parameters(model, parameters)
    val_losses, _, _, _, _, val_mae = validate(model, testloader)
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
# dwood_seed_2 = dwood + 'seed_67.pt'

# #Client_fn for fedprox
# def client_fn_fedprox(cid: str) -> FlowerClient:
#   """Create a Flower client representing a single organization."""
#
#   # Load model
#   net = load_model().to(DEVICE)
#
#   # Dataloaders is a dict with name as key and a tuple with trainloader and valloader as value
#   name = names[int(cid)]
#   trainloader, valloader = dataloaders[name]
#
#
#   # Create a  single FedProx representing a single organization
#   return FedProxClient(net, project_name, trainloader, valloader, cid, name)

# A function that returns a strategy and client_fn based on the strategy and save_dir
def get_config(strategy, save_dir, net, parameters):
  client_fn = gen_client_fn(project_name, strategy, save_dir)
  if strategy == 'FedAvg':
    return SaveFedAvg(
      fraction_fit=1.0,  # Sample 100% of available clients for training
      fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
      evaluate_fn=get_evaluate_fn(net, save_dir),
      on_fit_config_fn=fit_config,
      initial_parameters=parameters,
      save_dir=save_dir,
    ), client_fn
  elif strategy == 'FedProx':
    return SaveFedProx(
      fraction_fit=1.0,  # Sample 100% of available clients for training
      fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
      evaluate_fn=get_evaluate_fn(net, save_dir),
      on_fit_config_fn=fit_config,
      initial_parameters=parameters,
      proximal_mu=1.0,
      save_dir=save_dir,
    ), client_fn
# Create FedAvg strategy
# strategy = fl.server.strategy.FedAvg(
#     fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
#     evaluate_fn=get_evaluate_fn(net),
#     on_fit_config_fn=fit_config,  # Pass the fit_config function
#     initial_parameters=parameters,
#     # min_fit_clients=10,  # Never sample less than 10 clients for training
#     # min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
#     # min_available_clients=10,  # Wait until all 10 clients are available
# )
#
# fedavg = SaveFedAvg(
#   fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
#     evaluate_fn=get_evaluate_fn(net),
#     on_fit_config_fn=fit_config,
#   initial_parameters=parameters,
# )
#
# fedprox = SaveFedProx(
#   fraction_fit=1.0,  # Sample 100% of available clients for training
#     fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
#     evaluate_fn=get_evaluate_fn(net),
#     on_fit_config_fn=fit_config,
#   initial_parameters=parameters,
#   proximal_mu=1.0,
# )

# # Start simulation
# fl.simulation.start_simulation(
#     client_fn=client_fn,
#     num_clients=len(dfs),
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=fedavg,
#     client_resources=client_resources,
# )

# fl.simulation.start_simulation(
#     client_fn=client_fn_fedprox,
#     num_clients=len(dfs),
#     config=fl.server.ServerConfig(num_rounds=5),
#     strategy=fedprox,
#     client_resources=client_resources,
# )

#Generate a main function to run the simulation
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  #Argument for DWood seed
  parser.add_argument('--seed', type=str, required=False)
  #Argument for Federated Strategy
  parser.add_argument('--strategy', type=str, required=False)
  #FedAvg is default
  parser.set_defaults(strategy='FedAvg')
  args = parser.parse_args()
  #For the mode, if no seed provided, mode is RW
  if args.seed is None:
    mode = 'RW'
  else:
    mode = 'DWood'
  seed = f'_seed_{args.seed}' if args.seed is not None else ''
  project_name = f'{args.strategy}_{mode}_Dataset' + seed
  #Print the project name
  print(f'Now operating under project name {project_name}...')
  save_dir = './utils/models/' + project_name + "/"
  print(f'Saving models to {save_dir}...')
  # If the repository does not exist, create it
  if not os.path.exists(save_dir):
    print(f"Creating directory {save_dir}...")
    os.makedirs(save_dir)

  #Load the model
  #If a seed is defined, use it
  dwood_seed = dwood + f'seed_{args.seed}.pt'
  net = load_model(dwood_seed).to(DEVICE)

  weights = [val.cpu().numpy() for _, val in net.state_dict().items()]

  parameters = fl.common.ndarrays_to_parameters(weights)

  #get the client_fn and strategy from the arguments
  strategy, client_fn = get_config(args.strategy, save_dir, net, parameters)

  fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=len(dfs),
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
  )