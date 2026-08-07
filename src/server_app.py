import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import flwr as fl
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context

from centralized import load_model, get_test_loader, validate
from client import set_parameters
from strategy import SaveFedAvg, SaveFedProx
from utils import dwood, master_dataset
from datasets import load_dataset

def get_evaluate_fn(model, save_dir, testloader, device):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        print("Evaluating round", server_round)
        set_parameters(model, parameters)
        val_losses, _, _, _, _, val_mae = validate(model, testloader, device)
        # Write the losses to a file in save_dir
        with open(save_dir + 'centralized_losses.txt', 'a') as f:
            f.write(f"{server_round},{val_losses}\n")
        return float(val_losses), {"mae": float(val_mae)}
    return evaluate

def generate_fit_config(epochs: int, patience: int):
    def fit_config(server_round: int):
        config = {
            "server_round": server_round,
            "local_epochs": 1,
            "patience": patience
        }
        return config
    return fit_config

def server_fn(context: Context) -> ServerAppComponents:
    """Constructs the ServerAppComponents using the configuration in pyproject.toml."""
    # 1. Extract config from context
    run_config = context.run_config
    num_rounds = run_config.get("num-server-rounds", 5)
    strategy_name = run_config.get("strategy", "FedAvg")
    seed = run_config.get("seed", None)
    epochs = run_config.get("epochs", 20)
    patience = run_config.get("patience", 4)
    split = run_config.get("split", "dataset").capitalize()
    distribution = run_config.get("distribution", "Original")
    alias = run_config.get("alias", None)
    mock = run_config.get("mock", False)
    
    # 2. Setup project names and paths
    mode = 'RW' if not seed else 'DWood'
    seed_str = f'_seed_{seed}' if seed else ''
    alias_str = f'_{alias}' if alias is not None else ''
    if split.lower() == 'distribution':
        split += f'_{distribution}'
        
    project_name = f'{strategy_name}_{mode}_{split}{seed_str}{alias_str}'
    print(f'Now operating under project name {project_name}...')
    save_dir = f'./utils/models/{project_name}/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mode == 'DWood' and seed is not None:
        dwood_seed = os.path.join(dwood, f'seed_{seed}.pt')
        net = load_model(dwood_seed).to(device)
    else:
        net = load_model().to(device)
        
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(weights)

    # 4. Setup Global Evaluation Dataloader (Disabled for now)
    # raw_dataset = load_dataset("json", data_files=master_dataset)["train"]
    # splits = raw_dataset.train_test_split(test_size=0.2, seed=42)
    # testdf = splits["test"].to_pandas()
    # testloader = get_test_loader(testdf, batch_size=4, dataset_scale=1, mock=mock)

    # 5. Initialize Strategy
    if strategy_name == 'FedAvg':
        strategy = SaveFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            on_fit_config_fn=generate_fit_config(epochs, patience),
            initial_parameters=initial_parameters,
            save_dir=save_dir,
        )
    elif strategy_name == 'FedProx':
        strategy = SaveFedProx(
            fraction_fit=1.0,
            fraction_evaluate=0.5,
            on_fit_config_fn=generate_fit_config(epochs, patience),
            initial_parameters=initial_parameters,
            proximal_mu=1.0,
            save_dir=save_dir,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

# Create ServerApp
app = ServerApp(server_fn=server_fn)
