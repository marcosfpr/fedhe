import logging
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from sealy import (CkksEncryptionParametersBuilder, CoefficientModulus,
                   Context, DegreeType, SecurityLevel)

from fhelwr.strategy.fedavg import FedAvgSealy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [
        num_examples * float(m["accuracy"]) for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    degree = DegreeType(8192)
    security_level = SecurityLevel(128)
    bit_sizes = [60, 40, 40, 60]
    scale = 2**40

    expand_mod_chain = False
    modulus_chain = CoefficientModulus.create(degree, bit_sizes)
    encryption_parameters = (
        CkksEncryptionParametersBuilder()
        .with_poly_modulus_degree(degree)
        .with_coefficient_modulus(modulus_chain)
        .build()
    )
    context = Context.build(
        encryption_parameters, expand_mod_chain, security_level
    )

    # collected from the network used in the client.
    params_size = 3_972_122

    # Create FedAvg strategy
    strategy = FedAvgSealy(
        context,
        params_size,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
