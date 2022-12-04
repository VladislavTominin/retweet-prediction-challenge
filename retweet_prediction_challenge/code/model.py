import torch
import evaluate
import torch.nn as nn
import numpy as np
from transformers import CamembertModel
from transformers import CamembertTokenizer
from transformers.models.camembert.modeling_camembert import CamembertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback


class CamembertRegressor():

    def __init__(self, training_args):
        self.model_path = "camembert/camembert-base"
        self.model = CamembertForSequenceClassification.from_pretrained(self.model_path, num_labels=1) # 1 for Regression

        self.tokenizer = CamembertTokenizer.from_pretrained(self.model_path, model_max_length=512)
        self.tokenize_function = lambda examples: self.tokenizer(examples["text"], padding="max_length",
                                                                 truncation=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.training_args = TrainingArguments(**training_args)

    def get_tokenized_dataset(self, dataset):
        return dataset.map(self.tokenize_function, batched=True, num_proc=4)

    @staticmethod
    def compute_metrics(eval_preds, metric_list=None):
        if metric_list is None:
            metric_list = ["mse"]
        clf_metrics = evaluate.combine(metric_list)
        predictions, labels = eval_preds
        return clf_metrics.compute(predictions=predictions, references=labels)

    def train(self, dataset):
        tokenized_datasets = self.get_tokenized_dataset(dataset)
        trainer = Trainer(
            self.model,
            self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=self.compute_metrics,
            callbacks=[TensorBoardCallback()]
        )
        trainer.train()
        print(trainer.predict(tokenized_datasets["test"]))
