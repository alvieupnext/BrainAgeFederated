from flwr.common import NDArrays, Scalar, FitRes, Parameters
from typing import Dict, Optional, Tuple, List, Union, OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import FitIns,MetricsAggregationFn, NDArrays, Parameters, Scalar

from centralized import load_model, DEVICE
from client import set_parameters

net = load_model().to(DEVICE)

#New aggregate_fit method
def aggregate_fit(
        server_round: int,
        super_aggregate_fit,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        save_dir: str,

) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
  """Aggregate model weights using weighted average and store checkpoint"""

  # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
  aggregated_parameters, aggregated_metrics = super_aggregate_fit(server_round, results, failures)

  if aggregated_parameters is not None:
    print(f"Saving round {server_round} aggregated_parameters...")

    # Convert `Parameters` to `List[np.ndarray]`
    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

    # Set the parameters of the model
    set_parameters(net, aggregated_ndarrays)

    # Save the model
    torch.save(net.state_dict(), save_dir + f"federated_model_round_{server_round}.pt")

  return aggregated_parameters, aggregated_metrics


class SaveFedAvg(fl.server.strategy.FedAvg):
  # pylint: disable=too-many-arguments,too-many-instance-attributes
  def __init__(
          self,
          *,
          fraction_fit: float = 1.0,
          fraction_evaluate: float = 1.0,
          min_fit_clients: int = 2,
          min_evaluate_clients: int = 2,
          min_available_clients: int = 2,
          evaluate_fn: Optional[
            Callable[
              [int, NDArrays, Dict[str, Scalar]],
              Optional[Tuple[float, Dict[str, Scalar]]],
            ]
          ] = None,
          on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
          on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
          accept_failures: bool = True,
          initial_parameters: Optional[Parameters] = None,
          fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
          evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
          save_dir: str,
  ) -> None:
    super().__init__(
          fraction_fit=fraction_fit,
          fraction_evaluate=fraction_evaluate,
          min_fit_clients=min_fit_clients,
          min_evaluate_clients=min_evaluate_clients,
          min_available_clients=min_available_clients,
          evaluate_fn=evaluate_fn,
          on_fit_config_fn=on_fit_config_fn,
          on_evaluate_config_fn=on_evaluate_config_fn,
          accept_failures=accept_failures,
          initial_parameters=initial_parameters,
          fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
          evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
    self.save_dir = save_dir
  #
  def aggregate_fit(
          self,
          server_round: int,
          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
          failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate model weights using weighted average and store checkpoint"""

    # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
    aggregated_parameters, aggregated_metrics = aggregate_fit(server_round, super().aggregate_fit, results, failures, self.save_dir)

    return aggregated_parameters, aggregated_metrics

class SaveFedProx(fl.server.strategy.FedProx):
  # pylint: disable=too-many-arguments,too-many-instance-attributes
  def __init__(
          self,
          *,
          fraction_fit: float = 1.0,
          fraction_evaluate: float = 1.0,
          min_fit_clients: int = 2,
          min_evaluate_clients: int = 2,
          min_available_clients: int = 2,
          evaluate_fn: Optional[
            Callable[
              [int, NDArrays, Dict[str, Scalar]],
              Optional[Tuple[float, Dict[str, Scalar]]],
            ]
          ] = None,
          on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
          on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
          accept_failures: bool = True,
          initial_parameters: Optional[Parameters] = None,
          fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
          evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
          proximal_mu: float,
          save_dir: str,
  ) -> None:
    super().__init__(
          fraction_fit=fraction_fit,
          fraction_evaluate=fraction_evaluate,
          min_fit_clients=min_fit_clients,
          min_evaluate_clients=min_evaluate_clients,
          min_available_clients=min_available_clients,
          evaluate_fn=evaluate_fn,
          on_fit_config_fn=on_fit_config_fn,
          on_evaluate_config_fn=on_evaluate_config_fn,
          accept_failures=accept_failures,
          initial_parameters=initial_parameters,
          fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
          evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
          proximal_mu=proximal_mu,
        )
    self.save_dir = save_dir
  #
  def aggregate_fit(
          self,
          server_round: int,
          results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
          failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
  ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate model weights using weighted average and store checkpoint"""

    # Call aggregate_fit from base class (FedProx) to aggregate parameters and metrics
    aggregated_parameters, aggregated_metrics = aggregate_fit(server_round, super().aggregate_fit, results, failures, self.save_dir)
    return aggregated_parameters, aggregated_metrics


