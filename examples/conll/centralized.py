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


if __name__ == "__main__":
    mps_device = torch.device("mps")

    ner_data = datasets.load_dataset("conll2003")
    label_list = ner_data["train"].features["ner_tags"].feature.names  # type: ignore

    tokenizer = get_bert_tokenizer("bert-base-cased")
    data_collator = DataCollatorForTokenClassification(tokenizer)

    tokenized_data = ner_data.map(
        lambda batch: tokenize_and_align_labels(tokenizer, batch), batched=True
    )

    ner_model = get_tiny_bert_model_for_classification(9).to(mps_device)

    args = TrainingArguments(
        "centralized",
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
        logger_callback.train_metrics, "results/centralized_train_metrics.csv"
    )
    save_to_csv(
        logger_callback.eval_metrics, "results/centralized_eval_metrics.csv"
    )
