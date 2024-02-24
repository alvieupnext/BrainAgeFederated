from flwr.common import NDArrays, Scalar, FitRes, Parameters
from typing import Dict, Optional, Tuple, List, Union, OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.server.client_proxy import ClientProxy

from centralized import load_model, DEVICE
from client import set_parameters
from utils import save_dir

net = load_model().to(DEVICE)
class SaveModelStrategy(fl.server.strategy.FedAvg):
  def aggregate_fit(
          self,
          server_round: int,
          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
          failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate model weights using weighted average and store checkpoint"""

    # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
    aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

    if aggregated_parameters is not None:
      print(f"Saving round {server_round} aggregated_parameters...")

      # Convert `Parameters` to `List[np.ndarray]`
      aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

      # Set the parameters of the model
      set_parameters(net, aggregated_ndarrays)

      # Save the model
      torch.save(net.state_dict(), save_dir +  f"federated_model_round_{server_round}.pt")

    return aggregated_parameters, aggregated_metrics