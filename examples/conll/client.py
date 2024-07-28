import argparse
import logging
from typing import Dict, List, OrderedDict, Tuple

import datasets
import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import Scalar
from sealy import (BatchDecryptor, BatchEncryptor, CKKSBatchEncoder,
                   CkksEncryptionParametersBuilder, CoefficientModulus,
                   Context, DegreeType, SecurityLevel)
from sealy.sealy import PublicKey, SecretKey
from transformers import (DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

from fhelwr.client import SealyClient
from fhelwr.ext.ner.bert import (LogMetricsCallback, compute_metrics,
                                 get_bert_tokenizer,
                                 get_tiny_bert_model_for_classification,
                                 tokenize_and_align_labels)
from fhelwr.model import get_torch_number_params


class ConllClient(SealyClient):
    def __init__(
        self, cid: int, trainer: Trainer, keys_dir: str, test_data
    ) -> None:
        """
        Initialize the client with the neural network and data loaders.

        Args:
            cid: Client ID
        """
        super().__init__(cid)
        self.keys_dir = keys_dir
        self.trainer = trainer
        self.test_data = test_data

    def get_ctx(self) -> Context:
        """
        Return the SEALY context used by this client.
        """

        degree = DegreeType(8192)
        security_level = SecurityLevel(128)
        bit_sizes = [60, 40, 40, 60]

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
        return context

    def get_encoder(self):
        """
        Return the encryption scheme used by this client.
        """
        return CKKSBatchEncoder(self.get_ctx(), 2**40)

    def get_encryptor(self) -> BatchEncryptor:
        """
        Return the encryptor used by this client.
        """
        public_key = fetch_public_key(self.get_ctx(), self.keys_dir)
        return BatchEncryptor(self.get_ctx(), public_key)

    def get_decryptor(self) -> BatchDecryptor:
        """
        Return the decryptor used by this client.
        """
        secret_key = fetch_secret_key(self.get_ctx(), self.keys_dir)
        return BatchDecryptor(self.get_ctx(), secret_key)

    def set_params(self, parameters: List[np.ndarray]) -> None:
        """
        Load the parameters into the neural network.
        """
        logging.info(f"Setting parameters for client {self.cid}")
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        logging.debug(f"State dict: {state_dict}")
        self.trainer.model.load_state_dict(state_dict, strict=True)

    def get_params(self) -> List[np.ndarray]:
        """
        Get the parameters of the neural network.
        """
        logging.info(f"Getting parameters for client {self.cid}")
        # Get the state dictionary of the model
        state_dict = self.trainer.model.state_dict()
        # Convert each tensor in the state dictionary to a numpy array
        params = [param.cpu().numpy() for param in state_dict.values()]
        logging.debug(f"Parameters: {params}")
        return params

    def get_params_shape(self) -> List[np.ndarray]:
        """
        Get the shapes of the parameters of the neural network.
        """
        logging.info(f"Getting parameter shapes for client {self.cid}")
        # Get the state dictionary of the model
        state_dict = self.trainer.model.state_dict()
        # Get the shape of each parameter tensor in the state dictionary
        params_shape = [
            np.array(param.size()) for param in state_dict.values()
        ]
        logging.debug(f"Parameter shapes: {params_shape}")
        return params_shape

    def train(self) -> int:
        """
        Train the neural network.

        Returns:
            int: The number of examples used for training.
        """
        examples = len(self.trainer.get_train_dataloader())
        self.trainer.train()
        return examples

    def test(self) -> Tuple[int, float, Dict[str, Scalar]]:
        """
        Test the neural network.

        Returns:
            int: The number of examples used for testing.
            float: The loss.
            dict: The metrics.
        """
        metrics = self.trainer.evaluate(self.test_data)
        examples = self.test_data.num_rows
        loss = metrics["eval_loss"]
        return examples, loss, metrics  # type: ignore


def create_partitions(
    dataset,
    num_partitions: int,
    seed: int = 42,
):
    """
    Create partitioned versions of a source dataset with shuffling.

    Args:
        dataset (Dataset): The source dataset to partition.
        num_partitions (int): The number of partitions to create.
        seed (int): Random seed for shuffling.

    Returns:
        list: A list of datasets, each representing a partition.
    """
    # Shuffle the dataset with a random seed
    shuffled_dataset = dataset.shuffle(seed=seed)

    # Calculate the size of each partition
    partition_size = len(shuffled_dataset) // num_partitions

    partitions = []

    for i in range(num_partitions):
        # Define the start and end indices for the partition
        start_idx = i * partition_size
        end_idx = (
            (i + 1) * partition_size
            if i != num_partitions - 1
            else len(shuffled_dataset)
        )

        # Create the partition dataset
        partition = shuffled_dataset.select(range(start_idx, end_idx))
        partitions.append(partition)

    return partitions


def get_partition(client_id, total_clients, dataset):
    """
    Get the partition of the dataset for the client.

    Args:
        client_id (int): The client ID.
        total_clients (int): The total number of clients.
        dataset (Dataset): The source dataset.

    Returns:
        Dataset: The partition of the dataset for the client.
    """
    client_idx = client_id - 1

    train_partitions = create_partitions(dataset["train"], total_clients)
    test_partitions = create_partitions(dataset["test"], total_clients)
    validation_partitions = create_partitions(
        dataset["validation"], total_clients
    )

    logging.info(
        f"Number of training examples for client {client_id}:"
        f" {train_partitions[client_idx].num_rows}"
    )
    logging.info(
        f"Number of test examples for client {client_id}:"
        f" {test_partitions[client_idx].num_rows}"
    )
    logging.info(
        f"Number of validation examples for client {client_id}:"
        f" {validation_partitions[client_idx].num_rows}"
    )

    return {
        "train": train_partitions[client_idx],
        "test": test_partitions[client_idx],
        "validation": validation_partitions[client_idx],
    }


def fetch_public_key(ctx, keys_dir):
    with open(f"{keys_dir}/public_key", "rb") as f:
        public_key = PublicKey.from_bytes(ctx, f.read())
    return public_key


def fetch_secret_key(ctx, keys_dir):
    with open(f"{keys_dir}/secret_key", "rb") as f:
        secret_key = SecretKey.from_bytes(ctx, f.read())
    return secret_key


def save_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    mps_device = torch.device("mps")

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fhelwr").setLevel(logging.DEBUG)

    # receive from command-line args the client id, the ip address of the server, and the port number
    parser = argparse.ArgumentParser(description="SEALY Client")

    parser.add_argument("--client-id", type=int, help="Client ID")
    parser.add_argument("--server-ip", type=str, help="Server IP address")
    parser.add_argument("--server-port", type=int, help="Server port number")
    parser.add_argument("--keys-dir", type=str, help="Keys directory")
    parser.add_argument(
        "--total-clients", type=int, help="Total number of clients"
    )

    args = parser.parse_args()
    client_id = args.client_id
    server_ip = args.server_ip
    server_port = args.server_port
    keys_dir = args.keys_dir
    total_clients = args.total_clients

    ner_data = datasets.load_dataset("conll2003")

    client_data = get_partition(client_id, total_clients, ner_data)
    label_list = client_data["train"].features["ner_tags"].feature.names  # type: ignore

    tokenizer = get_bert_tokenizer("bert-base-cased")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tokenized_data = ner_data.map(
        lambda batch: tokenize_and_align_labels(tokenizer, batch), batched=True
    )

    ner_model = get_tiny_bert_model_for_classification(9).to(mps_device)

    args = TrainingArguments(
        f"client-{client_id}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        use_cpu=True,
    )

    def compute_metrics_impl(preds):
        return compute_metrics(label_list, preds)

    trainer = Trainer(
        ner_model,
        args,
        train_dataset=tokenized_data["train"],  # type: ignore
        eval_dataset=tokenized_data["validation"],  # type: ignore
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_impl,  # type: ignore
    )

    logger_callback = LogMetricsCallback(trainer)
    trainer.add_callback(logger_callback)

    print(f"number of parameters: {get_torch_number_params(trainer.model)})")
    print(f"trainable parameters: {trainer.get_num_trainable_parameters()}")

    client = ConllClient(client_id, trainer, keys_dir, tokenized_data["test"])  # type: ignore

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client,  # type: ignore
        grpc_max_message_length=2 * 536_870_912,
    )

    save_to_csv(
        logger_callback.train_metrics,
        f"results/federated/{client_id}_train_metrics.csv",
    )
    save_to_csv(
        logger_callback.eval_metrics,
        f"results/federated/{client_id}_eval_metrics.csv",
    )
    save_to_csv(
        client.performance_history(),
        f"results/federated/{client_id}_perf_metrics.csv",
    )
