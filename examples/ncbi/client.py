import argparse
import logging
import os
from typing import Dict, List, Tuple

import flwr as fl
import keras
import numpy as np
import pandas as pd
from flwr.common import Scalar
from flwr_datasets.federated_dataset import datasets
from flwr_datasets.utils import DatasetDict
from sealy import (BatchDecryptor, BatchEncryptor, CKKSBatchEncoder,
                   CkksEncryptionParametersBuilder, CoefficientModulus,
                   Context, DegreeType, SecurityLevel)
from sealy.sealy import PublicKey, SecretKey
from transformers import AutoTokenizer

from fhelwr.client import SealyClient

DEVICE = "cpu"
BATCH_SIZE = 32

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Used for type signatures
DATASET = Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]

XY = Tuple[np.ndarray, np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


def load_data() -> DatasetDict:
    """Load the data."""
    return datasets.load_dataset("ncbi_disease", revision="main")  # type: ignore


def shuffle(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx], mask[idx]


def partition(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, num_partitions: int
) -> XYList:
    """Split x and y into a number of partitions."""
    return list(
        zip(
            np.array_split(x, num_partitions),
            np.array_split(y, num_partitions),
            np.array_split(mask, num_partitions),
        )
    )


def create_partitions(
    source_dataset: XY,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a source dataset."""
    x, y, mask = source_dataset
    x, y, mask = shuffle(x, y, mask)
    xy_partitions = partition(x, y, mask, num_partitions)

    return xy_partitions


def get_dataset(dataset: DatasetDict, split: str) -> pd.DataFrame:
    dataframe = dataset[split].to_pandas()
    dataframe = dataframe[dataframe.ner_tags.str.len() > 0]  # pyright: ignore
    dataframe["num_tokens"] = dataframe.tokens.str.len()
    return dataframe  # pyright: ignore


def get_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


def tokenize(
    dataset: pd.DataFrame,
    max_length: int,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
):
    tokenizer = get_tokenizer()
    tokens = list(dataset.tokens.apply(lambda x: " ".join(x)).to_numpy())
    tokenized = tokenizer(
        tokens,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return tokenized


def normalize_tags(dataset, max_len):
    def make_tag_binary(tag, max_len):
        tag = np.array(tag)
        tag[tag > 1] = 1
        return np.pad(
            tag, pad_width=(0, max_len - len(tag))
        )  # the length of each token list has been padded to max_len, hence, tag list length has to match and has to be padded

    ner_tags = dataset.ner_tags.apply(lambda x: make_tag_binary(x, max_len))

    return ner_tags


def align_tags(tokens, tags):
    tokenized_inputs = tokens
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to 0
            if word_idx is None:
                label_ids.append(0)
            elif (
                word_idx != previous_word_idx
            ):  # Label all subwords with the same tag
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = np.array(labels)
    return tokenized_inputs


def get_tokenized_dataset(
    dataset: pd.DataFrame,
    max_length: int,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
):
    tokenized = tokenize(
        dataset, max_length, padding, truncation, return_tensors
    )
    normalized_tags = normalize_tags(dataset, max_length)
    aligned_tags = align_tags(tokenized, normalized_tags)
    x = np.array(tokenized["input_ids"]).squeeze()
    y = np.array(aligned_tags["labels"]).squeeze()
    mask = np.array(tokenized["attention_mask"]).squeeze()
    return x, y, mask


def sample_weights(y, class_weight):
    sample_weights = np.zeros(y.shape)
    for i, seq in enumerate(y):
        for j, label in enumerate(seq):
            sample_weights[i][j] = class_weight[label]
    return sample_weights


def get_dataset_partitions(num_clients):
    data = load_data()

    train_dataset = get_dataset(data, "train")
    test_dataset = get_dataset(data, "test")
    val_dataset = get_dataset(data, "validation")

    max_length: int = train_dataset["num_tokens"].max()  # pyright: ignore

    x_train, y_train, mask_train = get_tokenized_dataset(
        train_dataset, max_length=max_length
    )
    x_test, y_test, mask_test = get_tokenized_dataset(
        test_dataset, max_length=max_length
    )
    x_val, y_val, mask_val = get_tokenized_dataset(
        val_dataset, max_length=max_length
    )

    x_train_val, y_train_val, mask_train_val = (
        np.concatenate((x_train, x_val)),
        np.concatenate((y_train, y_val)),
        np.concatenate((mask_train, mask_val)),
    )

    train_partitions = create_partitions(
        (x_train_val, y_train_val, mask_train_val), num_clients
    )

    test_partitions = create_partitions(
        (x_test, y_test, mask_test), num_clients
    )

    return train_partitions, test_partitions


class FheClient(SealyClient):
    def __init__(self, cid, keys_dir, net, optimizer, loss_fn):
        super().__init__(cid)
        self.net = net  # tensorflow model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.keys_dir = keys_dir

    def with_train_dataset(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def with_test_dataset(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        return self

    def with_test_weights(self, test_weights):
        self.test_weights = test_weights
        return self

    def with_train_weights(self, train_weights):
        self.train_weights = train_weights
        return self

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

    def get_net(self):
        """
        Return the neural network used by this client.
        """
        return self.net

    def set_params(self, net, parameters: List[np.ndarray]) -> None:
        """
        Load the parameters into the neural network.
        """
        logging.info(f"Setting parameters for client {self.cid}")
        net.set_weights(parameters)

    def get_params(self, net) -> List[np.ndarray]:
        """
        Get the parameters of the neural network.
        """
        return net.get_weights()

    def get_params_shape(self, net) -> List[np.ndarray]:
        """
        Get the shapes of the parameters of the neural network.
        """
        return [w.shape for w in net.get_weights()]

    def train(self, net) -> int:
        """
        Train the neural network.

        Returns:
            int: The number of examples used for training.
        """
        examples = len(self.x_train)
        logging.info(f"Training client {self.cid} with {examples} examples...")
        net.fit(
            self.x_train,
            self.y_train,
            epochs=5,
        )
        logging.info(f"Training client {self.cid} done.")
        return examples

    def test(self, net) -> Tuple[int, float, Dict[str, Scalar]]:
        """
        Test the neural network.

        Returns:
            int: The number of examples used for testing.
            float: The loss.
            dict: The metrics.
        """
        logging.info(
            f"Testing client {self.cid} with {len(self.x_test)} examples..."
        )
        loss, accuracy = net.evaluate(
            self.x_test,
            self.y_test,
            batch_size=BATCH_SIZE,
            verbose=1,
            sample_weight=self.test_weights,
        )
        logging.info(f"Testing client {self.cid} done.")
        return len(self.x_test), float(loss), {"accuracy": float(accuracy)}


def fetch_public_key(ctx, keys_dir):
    with open(f"{keys_dir}/public_key", "rb") as f:
        public_key = PublicKey.from_bytes(ctx, f.read())
    return public_key


def fetch_secret_key(ctx, keys_dir):
    with open(f"{keys_dir}/secret_key", "rb") as f:
        secret_key = SecretKey.from_bytes(ctx, f.read())
    return secret_key


def get_rnn_model(
    tokenizer,
    embedding_dim=32,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
):
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                tokenizer.vocab_size,
                embedding_dim,
                mask_zero=True,
            ),
            keras.layers.Bidirectional(
                keras.layers.SimpleRNN(embedding_dim, return_sequences=True)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fhelwr").setLevel(logging.DEBUG)

    # receive from command-line args the client id, the ip address of the server, and the port number
    parser = argparse.ArgumentParser(description="SEALY CIFAR-10 Client")

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

    loss = keras.losses.SparseCategoricalCrossentropy()  # ignore_class=-100
    optimizer = keras.optimizers.Adam(learning_rate=0.0007)

    train_partitions, test_partitions = get_dataset_partitions(total_clients)

    x_train, y_train, mask_train = train_partitions[client_id]
    x_test, y_test, mask_test = test_partitions[client_id]

    class_weight = {0: 5.0, 1: 100.0}

    train_weights = sample_weights(y_train, class_weight)
    test_weights = sample_weights(y_test, class_weight)

    tokenizer = get_tokenizer()

    model = get_rnn_model(
        tokenizer,
        embedding_dim=128,
    )

    client = (
        FheClient(
            client_id,
            keys_dir,
            model,
            optimizer,
            loss,
        )
        .with_train_dataset(x_train, y_train)
        .with_test_dataset(x_test, y_test)
        .with_train_weights(train_weights)
        .with_test_weights(test_weights)
    )

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client,  # type: ignore
    )
