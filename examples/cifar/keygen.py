import argparse
import logging

import flwr as fl
from sealy import (CKKSBatchEncoder, CkksEncryptionParametersBuilder,
                   CoefficientModulus, Context, DegreeType, KeyGenerator,
                   SecurityLevel)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # receive from command-line args the client id, the ip address of the server, and the port number
    parser = argparse.ArgumentParser(description="SEALY Key Gen")
    parser.add_argument("--output-dir", type=str, help="Output Directory")

    args = parser.parse_args()
    output_dir = args.output_dir

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
    context = Context(encryption_parameters, expand_mod_chain, security_level)
    params_size = CKKSBatchEncoder(
        context, scale
    ).get_slot_count()  # pyright: ignore

    keygen = KeyGenerator(context)

    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    with open(f"{output_dir}/public_key", "wb") as f:
        f.write(bytes(public_key.as_bytes()))

    with open(f"{output_dir}/secret_key", "wb") as f:
        f.write(bytes(secret_key.as_bytes()))
