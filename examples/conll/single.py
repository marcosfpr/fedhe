import logging

import datasets
import pandas as pd
import torch
from transformers import (DataCollatorForTokenClassification, Trainer,
                          TrainingArguments)

from fhelwr.ext.ner.bert import (LogMetricsCallback, compute_metrics,
                                 get_bert_tokenizer,
                                 get_tiny_bert_model_for_classification,
                                 tokenize_and_align_labels)
from fhelwr.model import get_torch_number_params


def save_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)


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


if __name__ == "__main__":
    mps_device = torch.device("mps")

    ner_data = datasets.load_dataset("conll2003")

    client_data = get_partition(1, 10, ner_data)
    label_list = client_data["train"].features["ner_tags"].feature.names  # type: ignore

    tokenizer = get_bert_tokenizer("bert-base-cased")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tokenized_data = ner_data.map(
        lambda batch: tokenize_and_align_labels(tokenizer, batch), batched=True
    )

    ner_model = get_tiny_bert_model_for_classification(9).to(mps_device)

    args = TrainingArguments(
        "single",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
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

    trainer.train()

    save_to_csv(
        logger_callback.train_metrics, "results/single_train_metrics.csv"
    )
    save_to_csv(
        logger_callback.eval_metrics, "results/single_eval_metrics.csv"
    )
