import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import flwr as fl
import numpy as np
from flwr.common import (Code, EvaluateIns, EvaluateRes, FitIns, FitRes,
                         GetParametersIns, GetParametersRes, Parameters,
                         Scalar, Status)
from sealy import BatchDecryptor, BatchEncryptor, CiphertextBatchArray, Context

from fhelwr.model import (ciphertext_to_params, flatten_parameters,
                          params_to_ciphertext, unflatten_parameters)


class SealyClient(fl.client.Client, ABC):
    """
    Flower client that uses SEALY to encrypt the model parameters.
    """

    def __init__(
        self,
        cid: int,
    ) -> None:
        """
        Initialize the client with the neural network and data loaders.

        Args:
            cid: Client ID
        """
        self.cid = cid
        self.encrypt_history = []
        self.encode_history = []
        self.decode_history = []
        self.decrypt_history = []
        self.bytes_sent = []

    @abstractmethod
    def get_ctx(self) -> Context:
        """
        Return the SEALY context used by this client.
        """
        pass

    @abstractmethod
    def get_encoder(self) -> Any:
        """
        Return the encryption scheme used by this client.
        """
        pass

    @abstractmethod
    def get_encryptor(self) -> BatchEncryptor:
        """
        Return the encryptor used by this client.
        """
        pass

    @abstractmethod
    def get_decryptor(self) -> BatchDecryptor:
        """
        Return the decryptor used by this client.
        """
        pass

    @abstractmethod
    def set_params(self, parameters: List[np.ndarray]) -> None:
        """
        Load the parameters into the neural network.
        """
        pass

    @abstractmethod
    def get_params(self) -> List[np.ndarray]:
        """
        Get the parameters of the neural network.
        """
        pass

    @abstractmethod
    def get_params_shape(self) -> List[np.ndarray]:
        """
        Get the shapes of the parameters of the neural network.
        """
        pass

    @abstractmethod
    def train(self) -> int:
        """
        Train the neural network.

        Returns:
            int: The number of examples used for training.
        """
        pass

    @abstractmethod
    def test(self) -> Tuple[int, float, Dict[str, Scalar]]:
        """
        Test the neural network.

        Returns:
            int: The number of examples used for testing.
            float: The loss.
            dict: The metrics.
        """
        pass

    def __encrypt_params(self, parameters: np.ndarray) -> CiphertextBatchArray:
        """
        Get the parameters of the neural network as an encrypted tensor.
        """
        start = time.time()
        encoded = self.get_encoder().encode(parameters.tolist())
        end = time.time()
        self.encode_history.append(end - start)

        start = time.time()
        encrypted = self.get_encryptor().encrypt(encoded)
        end = time.time()
        self.encrypt_history.append(end - start)

        return encrypted

    def __decrypt_params(self, encrypted: CiphertextBatchArray) -> np.ndarray:
        start = time.time()
        decrypted = self.get_decryptor().decrypt(encrypted)
        end = time.time()
        self.decrypt_history.append(end - start)

        start = time.time()
        decoded = np.asarray(self.get_encoder().decode(decrypted))
        end = time.time()
        self.decode_history.append(end - start)

        return decoded.flatten()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        logging.info("Client %d: get_parameters", self.cid)

        params = self.get_params()

        if not params:
            return GetParametersRes(
                parameters=fl.common.Parameters(
                    tensors=[], tensor_type="CiphertextBatchArray"
                ),
                status=Status(code=Code.OK, message="No parameters found"),
            )

        net_parameters = flatten_parameters(params)

        # Get parameters as encrypted tensor
        logging.debug("Encrypting parameters for client %d", self.cid)
        encrypted_parameters = self.__encrypt_params(net_parameters)
        logging.debug("Encrypted parameters for client %d", self.cid)

        # Serialize to bytes
        logging.debug(
            "Serializing encrypted parameters for client %d", self.cid
        )
        parameters = ciphertext_to_params(encrypted_parameters)
        logging.debug(
            "Serialized encrypted parameters for client %d", self.cid
        )

        # Status
        status = Status(code=Code.OK, message="Success")

        self.bytes_sent.append(self.get_params_size_bytes(parameters))

        return GetParametersRes(parameters=parameters, status=status)

    def fit(self, ins: FitIns) -> FitRes:
        logging.info("Client %d: fit", self.cid)

        ciphered_parameters = ins.parameters

        logging.info(
            "Received %d parameters of type %s",
            len(ciphered_parameters.tensors),
            ciphered_parameters.tensor_type,
        )

        parameters = params_to_ciphertext(self.get_ctx(), ciphered_parameters)
        logging.debug("Decrypting parameters for client %d", self.cid)
        decrypted_parameters = self.__decrypt_params(parameters)
        logging.debug("Decrypted parameters for client %d", self.cid)

        # Unflatten parameters
        shapes = self.get_params_shape()
        unflat_params = unflatten_parameters(shapes, decrypted_parameters)
        # Set parameters
        logging.debug("Setting parameters for client %d", self.cid)
        self.set_params(unflat_params)

        # Train
        logging.debug("Training for client %d", self.cid)
        num_examples = self.train()
        logging.debug("Trained for client %d", self.cid)

        # Get updated parameters
        logging.debug("Getting parameters for client %d", self.cid)
        params = self.get_params()
        updated_parameters = flatten_parameters(params)
        logging.debug("Got parameters for client %d", self.cid)

        # Encrypt parameters
        logging.debug("Encrypting parameters for client %d", self.cid)
        encrypted_parameters = self.__encrypt_params(updated_parameters)
        logging.debug("Encrypted parameters for client %d", self.cid)

        # Serialize to bytes
        logging.debug(
            "Serializing encrypted parameters for client %d", self.cid
        )
        parameters = ciphertext_to_params(encrypted_parameters)
        logging.debug(
            "Serialized %d encrypted parameters for client %d",
            len(parameters.tensors),
            self.cid,
        )

        # Status
        status = Status(code=Code.OK, message="Success")

        return FitRes(
            parameters=parameters,
            num_examples=num_examples,
            status=status,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        logging.info("Client %d: evaluate", self.cid)

        ciphered_parameters = ins.parameters

        logging.info(
            "Received %d parameters of type %s",
            len(ciphered_parameters.tensors),
            ciphered_parameters.tensor_type,
        )

        # Decrypt parameters
        parameters = params_to_ciphertext(self.get_ctx(), ciphered_parameters)
        logging.debug("Decrypting parameters for client %d", self.cid)
        decrypted_parameters = self.__decrypt_params(parameters)
        logging.debug("Decrypted parameters for client %d", self.cid)

        # Unflatten parameters
        shapes = self.get_params_shape()
        unflat_params = unflatten_parameters(shapes, decrypted_parameters)

        # Set parameters
        logging.debug("Setting parameters for client %d", self.cid)
        self.set_params(unflat_params)
        logging.debug("Set parameters for client %d", self.cid)

        logging.debug("Testing for client %d", self.cid)
        num_examples, loss, metrics = self.test()
        logging.debug("Tested for client %d", self.cid)

        # Status
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            status=status,
            metrics=metrics,
        )

    def performance_history(self) -> Dict[str, List[float]]:
        return {
            "encrypt": self.encrypt_history,
            "encode": self.encode_history,
            "decode": self.decode_history,
            "decrypt": self.decrypt_history,
            "bytes_sent": self.bytes_sent,
        }

    def get_params_size_bytes(self, params: Parameters) -> int:
        """
        Get the size of the parameters in bytes.
        """
        return sum([len(param) for param in params.tensors])
