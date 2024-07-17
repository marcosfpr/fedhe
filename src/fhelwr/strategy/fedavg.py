import logging
from collections.abc import Callable
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (EvaluateIns, EvaluateRes, FitIns, FitRes,
                         MetricsAggregationFn, Parameters, Scalar)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from sealy import (CiphertextBatchArray, CKKSBatchEncoder, CKKSBatchEvaluator,
                   Context)

from fhelwr.model import ciphertext_to_params, params_to_ciphertext


def aggregate_ciphertext(
    context: Context,
    params: List[Tuple[CiphertextBatchArray, int]],
    size_params: int,
) -> CiphertextBatchArray:
    """
    Aggregate a list of CiphertextBatchArray objects using weighted average.

    Args:
    params: List of tuples, where each tuple contains a CiphertextBatchArray
    and an integer corresponding the number of examples used to compute the
    CiphertextBatchArray.

    """

    evaluator = CKKSBatchEvaluator(context)
    encoder = CKKSBatchEncoder(context, 2**40)

    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in params)

    # Create a list of weights, each multiplied by the related number of examples
    weights = [
        1.0 * num_examples / num_examples_total for (_, num_examples) in params
    ]

    # Add the CiphertextBatchArray objects using the weighted weights
    weighted_ciphertexts = [
        evaluator.multiply_plain(
            ciphertext, encoder.encode(size_params * [weight])
        )
        for (ciphertext, weight) in zip(
            [ciphertext for (ciphertext, _) in params], weights
        )
    ]

    result = evaluator.add_many(weighted_ciphertexts)

    return result


class FedAvgSealy(fl.server.strategy.Strategy):
    """
    Base class for Federated Learning with SEALY.
    """

    def __init__(
        self,
        context: Context,
        params_size: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, CiphertextBatchArray, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[
            Callable[[int], Dict[str, Scalar]]
        ] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.context = context
        self.params_size = params_size
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        return "FedSealy"

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        logging.info(f"num_fit_clients: {num_available_clients}")
        num_clients = int(num_available_clients * self.fraction_fit)
        return (
            max(num_clients, self.min_fit_clients),
            self.min_available_clients,
        )

    def num_evaluation_clients(
        self, num_available_clients: int
    ) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        logging.info(f"num_evaluation_clients: {num_available_clients}")
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return (
            max(num_clients, self.min_evaluate_clients),
            self.min_available_clients,
        )

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        logging.info("initialize_parameters")
        initial_parameters = self.initial_parameters
        self.initial_parameters = (
            None  # Don't keep initial parameters in memory
        )
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        logging.info(f"evaluate: {server_round}")
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        cipher_params = params_to_ciphertext(self.context, parameters)
        eval_res = self.evaluate_fn(server_round, cipher_params, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        logging.info(f"configure_fit: {server_round}")

        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        logging.info(f"configure_evaluate: {server_round}")
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        logging.info(f"aggregate_fit: {server_round}")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = []

        for client, fit_res in results:
            cipher_params = params_to_ciphertext(
                self.context, fit_res.parameters
            )
            weights_results.append((cipher_params, fit_res.num_examples))
            logging.info(
                f"Client {client.cid} has {fit_res.num_examples} examples     "
                "            and parameters in"
                f" {len(fit_res.parameters.tensors)} chunks"
            )

        logging.info(
            f"Aggregating parameters from {len(weights_results)} clients..."
        )
        aggregated_ndarrays = aggregate_ciphertext(
            self.context, weights_results, self.params_size
        )

        parameters_aggregated = ciphertext_to_params(aggregated_ndarrays)
        logging.info(
            "Aggregated parameters. Total chunks: %d",
            len(parameters_aggregated.tensors),
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            logging.warning("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        logging.info(f"aggregate_evaluate: {server_round}")
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [
                (res.num_examples, res.metrics) for _, res in results
            ]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                eval_metrics
            )
        elif server_round == 1:  # Only log this warning once
            logging.warning("No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
