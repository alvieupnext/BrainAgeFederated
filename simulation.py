from centralized import load_model, DEVICE, get_test_loader, group_datasets, get_train_valid_loader, validate
from client import FlowerClient, set_parameters
import pandas as pd
import flwr as fl
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
# Load patients_dataset_6326_train.csv
df = pd.read_csv('patients_dataset_6326_train.csv')
#Group the dataframe in different dataframes by dataset attribute
dfs = group_datasets(df, mode='100')
# #Remove PDD from the dictionary
# dfs.pop('PDD')
dataloaders = {name: get_train_valid_loader(df, batch_size=3, random_seed=10) for name, df in dfs.items()}
names = list(dataloaders.keys())
print(names)
print(dataloaders)
# #Print the dataframe from Other and from PDD
# print(dfs.get('PDD'))
def client_fn(cid: str) -> FlowerClient:
  """Create a Flower client representing a single organization."""

  # Load model
  net = load_model().to(DEVICE)

  # Dataloaders is a dict with name as key and a tuple with trainloader and valloader as value
  name = names[int(cid)]
  trainloader, valloader = dataloaders[name]


  # Create a  single Flower client representing a single organization
  return FlowerClient(net, trainloader, valloader, cid, name)

#Evaluation server side using test csv
def get_evaluate_fn(model):
  testdf = pd.read_csv('patients_dataset_6326_test.csv')
  testloader = get_test_loader(testdf, batch_size=4)
  print("Loaded test data")
  def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    print("Evaluating round", server_round)
    set_parameters(model, parameters)
    val_losses, _, _, _, _, val_mae = validate(model, testloader)
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

net = load_model().to(DEVICE)

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    evaluate_fn=get_evaluate_fn(net)
    # min_fit_clients=10,  # Never sample less than 10 clients for training
    # min_evaluate_clients=2,  # Never sample less than 5 clients for evaluation
    # min_available_clients=10,  # Wait until all 10 clients are available
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
    client_resources=client_resources,
)