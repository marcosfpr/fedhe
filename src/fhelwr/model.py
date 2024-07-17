from typing import List

import numpy as np
from datasets.utils.logging import logging
from flwr.common import Parameters
from sealy import CiphertextBatchArray, Context


def ciphertext_to_params(ciphertext: CiphertextBatchArray) -> Parameters:
    """
    Convert a ciphertext to a Parameters object.
    """
    batched_bytes = [bytes(chunk) for chunk in ciphertext.as_batched_bytes()]
    return Parameters(
        tensors=batched_bytes, tensor_type="CiphertextBatchArray"
    )


def params_to_ciphertext(
    context: Context, params: Parameters
) -> CiphertextBatchArray:
    """
    Convert a Parameters object to a ciphertext.
    """
    cipher = CiphertextBatchArray.from_batched_bytes(context, params.tensors)
    return cipher


def flatten_parameters(net) -> np.ndarray:
    """
    Flatten all parameters of the model into a single 1D NumPy array.

    Args:
    net (torch.nn.Module): The PyTorch model.

    Returns:
    np.ndarray: A 1D NumPy array containing all model parameters.
    """
    logging.info(f"Flattening model parameters...")
    params = [
        param.detach().cpu().numpy().flatten() for param in net.parameters()
    ]
    flat_params = np.concatenate(params)
    return flat_params


def unflatten_parameters(net, flatten_params: np.ndarray) -> List[np.ndarray]:
    """
    Unflatten the 1D NumPy array back to the original parameter shapes of the model.

    Args:
    net (torch.nn.Module): The PyTorch model.
    flatten_params (np.ndarray): The 1D NumPy array containing all flattened model parameters.

    Returns:
    List[np.ndarray]: A list of NumPy arrays with the original shapes of the model parameters.
    """
    total_params = flatten_params.shape[0]
    param_shapes = [param.shape for param in net.parameters()]

    logging.info(
        f"Unflattening {total_params} parameters with shapes: {param_shapes}"
    )

    unflat_params = []
    start = 0

    for shape in param_shapes:
        num_elements = np.prod(shape)
        param_flat = flatten_params[start : start + num_elements]
        param = param_flat.reshape(shape)
        unflat_params.append(param)
        start += num_elements

    return unflat_params
