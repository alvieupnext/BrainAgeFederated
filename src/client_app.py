import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
from datasets import load_dataset
from flwr.client import ClientApp
from flwr.common import Context

from client import FlowerClient, FedProxClient
from centralized import load_model
from utils import master_dataset
from partitioner import BrainAgePartitioner

def client_fn(context: Context):
    # 1. Read config from context
    run_config = context.run_config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    strategy_name = run_config.get("strategy", "FedAvg")
    seed = run_config.get("seed", None)
    split = run_config.get("split", "dataset").capitalize()
    distribution = run_config.get("distribution", "Original")
    alias = run_config.get("alias", None)
    kcrossval = run_config.get("kcrossval", 10)
    mock = run_config.get("mock", False)
    
    # 2. Setup project name and save dir
    mode = 'RW' if not seed else 'DWood'
    seed_str = f'_seed_{seed}' if seed else ''
    alias_str = f'_{alias}' if alias is not None else ''
    if split.lower() == 'distribution':
        split += f'_{distribution}'
        
    project_name = f'{strategy_name}_{mode}_{split}{seed_str}{alias_str}'
    save_dir = f'./utils/models/{project_name}/'
    
    # 3. Load dataset locally and grab this node's specific partition
    raw_dataset = load_dataset("json", data_files=master_dataset)["train"]
    dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)["train"]
    
    partitioner = BrainAgePartitioner(distribution_type=distribution, num_partitions=num_partitions)
    partitioner.dataset = dataset
    
    partition = partitioner.load_partition(partition_id)
    # Convert back to Pandas for your existing DataLoaders
    partition_df = partition.to_pandas()
    
    # 4. Device and Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model().to(device)
    
    # 5. Initialize the client
    # Instead of pulling names from a pre-calculated dictionary, we just name the node dynamically
    name = f"{distribution}_{partition_id}"
    
    if strategy_name == 'FedAvg':
        # to_client() converts NumPyClient to standard Flower Client
        return FlowerClient(net, project_name, save_dir, partition_df, str(partition_id), name=name, kcrossval=kcrossval, device=device, mock=mock).to_client()
    elif strategy_name == 'FedProx':
        return FedProxClient(net, project_name, save_dir, partition_df, str(partition_id), name=name, kcrossval=kcrossval, device=device, mock=mock).to_client()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

app = ClientApp(client_fn=client_fn)
