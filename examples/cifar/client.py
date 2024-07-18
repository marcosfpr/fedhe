import argparse
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from flwr.common import Scalar
from flwr_datasets import FederatedDataset
from sealy import (BatchDecryptor, BatchEncryptor, CKKSBatchEncoder,
                   CkksEncryptionParametersBuilder, CoefficientModulus,
                   Context, DegreeType, SecurityLevel)
from sealy.sealy import PublicKey, SecretKey
from torch.utils.data import DataLoader

from fhelwr.client import SealyClient

NUM_CLIENTS = 5
BATCH_SIZE = 32


USE_FEDBN: bool = True
DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower"
    f" {fl.__version__}"
)
disable_progress_bar()


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs: int, verbose=True, cid: int = -1):
    """Train the network on the training set."""
    logging.info(f"Training client {cid} for {epochs} epochs")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        logging.info(f"[Client {cid}] Epoch {epoch+1}/{epochs}")
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            logging.info(
                f"[Client {cid}] Epoch {epoch+1}: train loss {epoch_loss},"
                f" accuracy {epoch_acc}"
            )


def test(net, testloader, cid: int = -1):
    """Evaluate the network on the entire test set."""
    logging.info(f"Testing {cid}...")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, total_partitions: int = 10):
    """Load partition CIFAR10 data."""
    fds = FederatedDataset(
        dataset="cifar10", partitioners={"train": total_partitions}
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(
        apply_transforms
    )
    trainloader = DataLoader(
        partition_train_test["train"],  # pyright: ignore
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=BATCH_SIZE  # pyright: ignore
    )
    return trainloader, testloader


class FheClient(SealyClient):
    def __init__(
        self,
        cid,
        net,
        trainloader,
        valloader,
        epochs=10,
    ):
        super().__init__(cid)
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epochs = epochs

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
        public_key = fetch_public_key(self.get_ctx())
        return BatchEncryptor(self.get_ctx(), public_key)

    def get_decryptor(self) -> BatchDecryptor:
        """
        Return the decryptor used by this client.
        """
        secret_key = fetch_secret_key(self.get_ctx())
        return BatchDecryptor(self.get_ctx(), secret_key)

    def get_net(self):
        """
        Return the neural network used by this client.
        """
        return self.net

    def set_parameters(self, net, parameters: List[np.ndarray]) -> None:
        """
        Load the parameters into the neural network.
        """
        logging.info(f"Setting parameters for client {self.cid}")
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        logging.debug(f"State dict: {state_dict}")
        net.load_state_dict(state_dict, strict=True)

    def train(self, net) -> int:
        """
        Train the neural network.

        Returns:
            int: The number of examples used for training.
        """
        examples = len(self.trainloader)
        logging.info(f"Training client {self.cid} with {examples} examples...")
        train(net, self.trainloader, epochs=self.epochs, cid=self.cid)
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
            f"Testing client {self.cid} with {len(self.valloader)} examples..."
        )
        loss, accuracy = test(net, self.valloader, cid=self.cid)
        logging.info(f"Testing client {self.cid} done.")
        return len(self.valloader), float(loss), {"accuracy": float(accuracy)}


def fetch_public_key(ctx):
    with open(f"{keys_dir}/public_key", "rb") as f:
        public_key = PublicKey.from_bytes(ctx, f.read())
    return public_key


def fetch_secret_key(ctx):
    with open(f"{keys_dir}/secret_key", "rb") as f:
        secret_key = SecretKey.from_bytes(ctx, f.read())
    return secret_key


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fhelwr").setLevel(logging.DEBUG)

    # receive from command-line args the client id, the ip address of the server, and the port number
    parser = argparse.ArgumentParser(description="SEALY CIFAR-10 Client")

    parser.add_argument("--client-id", type=int, help="Client ID")
    parser.add_argument("--server-ip", type=str, help="Server IP address")
    parser.add_argument("--server-port", type=int, help="Server port number")
    parser.add_argument("--keys-dir", type=str, help="Keys directory")

    args = parser.parse_args()
    client_id = args.client_id
    server_ip = args.server_ip
    server_port = args.server_port
    keys_dir = args.keys_dir

    trainloader, valloader = load_data(client_id, NUM_CLIENTS)

    model = Net().to(DEVICE)
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    client = FheClient(
        client_id,
        model,
        trainloader,
        valloader,
    )

    fl.client.start_client(
        server_address=f"{server_ip}:{server_port}",
        client=client,  # type: ignore
    )
