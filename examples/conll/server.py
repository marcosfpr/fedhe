import logging
from typing import List, Tuple

import flwr as fl
import pandas as pd
import torch
from flwr.common import Metrics
from sealy import (CkksEncryptionParametersBuilder, CoefficientModulus,
                   Context, DegreeType, SecurityLevel)
from transformers import Trainer, TrainingArguments

from fhelwr.ext.ner.bert import get_tiny_bert_model_for_classification
from fhelwr.strategy.fedavg import FedAvgSealy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    try:
        accuracies = [
            num_examples * float(m["eval_accuracy"])
            for num_examples, m in metrics
        ]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"accuracy": sum(accuracies) / sum(examples)}
    except KeyError as e:
        logging.error(
            "KeyError in weighted_average: 'eval_accuracy' not found"
        )
        logging.error("List of metrics: %s", metrics)
        raise e


def get_shared_model():
    mps_device = torch.device("mps")
    ner_model = get_tiny_bert_model_for_classification(9).to(mps_device)

    args = TrainingArguments(
        "coordinator",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        use_cpu=True,
    )

    return Trainer(ner_model, args)


def save_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)


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

    model = get_shared_model()
    params_size = model.get_num_trainable_parameters()
    logging.info(f"Number of parameters: {params_size}")

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
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        grpc_max_message_length=2 * 536_870_912,
    )

    save_to_csv(
        strategy.performance_history(),
        f"results/federated/coordinator_perf_metrics.csv",
    )
