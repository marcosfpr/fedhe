import datasets
import numpy as np
from datasets.arrow_dataset import deepcopy
from transformers import (AutoModelForTokenClassification, BertTokenizerFast,
                          TrainerCallback)


def get_bert_tokenizer(model_name):
    return BertTokenizerFast.from_pretrained(model_name)


def get_tiny_bert_model_for_classification(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        "prajjwal1/bert-tiny", num_labels=num_labels
    )


def tokenize_and_align_labels(tokenizer, examples, label_all_tokens=True):
    """
    Tokenize all words in the examples and align the labels with the tokenized inputs.

    Args:
        tokenizer: The tokenizer to use.

        examples: The examples to tokenize. It's a dictionary with two keys: "tokens" and "ner_tags".
        The tokens are the word list and the ner_tags are the corresponding NER tags.

        label_all_tokens: Whether or not to set the labels of all tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # word_ids() => Return a list mapping the tokens
        # to their actual word in the initial sentence.
        # It Returns a list indicating the word corresponding to each token.
        previous_word_idx = None
        label_ids = []
        # Special tokens like `` and `<\s>` are originally mapped to None
        # We need to set the label to -100 so they are automatically ignored in the loss function.
        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                # if current word_idx is != prev then its the most regular case
                # and add the corresponding token
                label_ids.append(label[word_idx])
            else:
                # to take care of sub-words which have the same word_idx
                # set -100 as well for them, but only if label_all_tokens == False
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def print_tokenized_example(tokenizer, word_ids, labels):
    for token, label in zip(tokenizer.convert_ids_to_tokens(word_ids), labels):
        print(f"{token:_<40} {label}")


def compute_metrics(label_list, eval_preds):
    metric = datasets.load_metric("seqeval")

    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [
            label_list[eval_preds]
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
        [label_list[l] for (_, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)  # type: ignore

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class LogMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer
        self.eval_metrics = []
        self.train_metrics = []

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset,
                metric_key_prefix="train",
            )
            self._trainer.evaluate(
                eval_dataset=self._trainer.eval_dataset,
                metric_key_prefix="eval",
            )

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", {})
        epoch = state.epoch
        metrics["epoch"] = epoch
        if "eval_loss" in metrics:
            self.eval_metrics.append(metrics)
        elif "train_loss" in metrics:
            self.train_metrics.append(metrics)
        else:
            raise ValueError("No valid metrics")
