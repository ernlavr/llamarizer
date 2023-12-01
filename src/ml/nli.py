""" Developed by Viktor Due Pedersen

This file contains the code for the Natural Language Inference (NLI) model.

Finetuned on the X-Sum Factuality dataset from huggingface:
    https://huggingface.co/datasets/xsum_factuality

The model is based on AutoModelForSequenceClassification from huggingface:
    https://huggingface.co/docs/transformers/model_doc/auto

Developed on DistilBert, but can be changed to any model from huggingface:
    https://huggingface.co/docs/transformers/model_doc/distilbert

"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import EvalPrediction

from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict

# .env file contains the WANDB_API_KEY and WANDB_PROJECT
load_dotenv()

# set wandb to offline mode
os.environ["WANDB_MODE"] = "offline"


@dataclass
class NLI_Finetune:
    HF_MODEL_NAME: str = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    dataset_factuality: DatasetDict = field(
        default_factory=lambda: load_dataset("xsum_factuality")
    )
    dataset_xsum: DatasetDict = field(
        default_factory=lambda: load_dataset(
            "EdinburghNLP/xsum", split="train+validation+test"
        )
    )
    merged_dataset: Dataset = None
    dataset: DatasetDict = None

    def __post_init__(self):
        def merge_datasets():
            """Should merge the X-sum dataset with the X-sum factuality dataset based on the bbcid. The two datasets have
            different columns, so the merge should be done on the bbcid column.

            dataset_factuality:
                bbcid: Document id in the XSum corpus.
                system: Name of neural summarizer.
                summary: Summary generated by ‘system’.
                is_factual: Yes (1) or No (0)
                worker_id: Worker ID (one of 'wid_0', 'wid_1', 'wid_2')

            dataset_xsum:
                document: Input news article.
                summary: One sentence summary of the article.
                id: BBC ID of the article.
            """

            # rename the bbcid column to id
            self.dataset_factuality = self.dataset_factuality.rename_column(
                "bbcid", "id"
            )

            """
            dataset_factuality:
                DatasetDict({
                    train: Dataset({
                        features: ['id', 'system', 'summary', 'is_factual', 'worker_id'],
                        num_rows: 5597 }) })

            dataset_xsum:
                Dataset({
                    features: ['document', 'summary', 'id'],
                    num_rows: 226711 })
            """

            # Convert the xsum dataset to a dictionary for quick lookup
            xsum_dict: Dict[int, Dict] = {
                int(example["id"]): example for example in self.dataset_xsum
            }

            # Prepare a new dataset
            merged_data = {
                "id": [],
                "document": [],
                "summary": [],
                "is_factual": [],
            }

            # Loop through the factuality dataset and merge
            for example in self.dataset_factuality["train"]:
                factuality_id: int = example["id"]
                xsum_example = xsum_dict.get(factuality_id)
                if xsum_example:
                    merged_data["id"].append(example["id"])
                    merged_data["document"].append(xsum_example["document"])
                    merged_data["summary"].append(xsum_example["summary"])
                    merged_data["is_factual"].append(example["is_factual"])

            self.merged_dataset = Dataset.from_dict(merged_data)

            """
            merged_dataset:
                Dataset({
                    features: ['id', 'document', 'summary', 'is_factual'],
                    num_rows: 5597 })
            """

        def split_dataset():
            """Since the X-sum dataset only contains a train set, we split it into train and validation set.
            Note that the texts are replicated three times and should be grouped by "bbcid
            """
            bbcids: Set[str] = set(self.merged_dataset["id"])

            # split the bbcids into train and test
            train_bbcids = set(list(bbcids)[: int(len(bbcids) * 0.8)])
            test_bbcids = bbcids - train_bbcids

            # filter the dataset

            self.dataset = DatasetDict()

            self.dataset["test"] = self.merged_dataset.filter(
                lambda example: example["id"] in test_bbcids
            )
            self.dataset["train"] = self.merged_dataset.filter(
                lambda example: example["id"] in train_bbcids
            )

            # make sure that there don't exist a bbcid in both train and test
            train_bbcids = set(self.dataset["train"]["id"])
            test_bbcids = set(self.dataset["test"]["id"])
            assert (
                len(train_bbcids.intersection(test_bbcids)) == 0
            ), "The same id exists in both train and test"

        self.config = AutoConfig.from_pretrained(self.HF_MODEL_NAME, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.HF_MODEL_NAME, config=self.config
        )

        merge_datasets()

        # The is_factual column has NULL value for some entries which have been replaced iwth -1. These are removed.
        self.merged_dataset = self.merged_dataset.filter(
            lambda example: example["is_factual"] != -1
        )

        # split into train and test
        split_dataset()

    def compute_metrics(self, pred: EvalPrediction):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def finetune(self):
        def tokenize_function(examples: Dict) -> BatchEncoding:
            # `id: Document id in the XSum corpus.
            id: List[int] = examples["id"]
            # document:
            document: List[str] = examples["document"]
            # summary: Summary generated by ‘system’.
            summary: List[str] = examples["summary"]
            # is_factual: Yes (1) or No (0)
            is_factual: List[int] = examples["is_factual"]  # 0, 1

            # Tokenize all texts and align the labels with them.
            tokenized_inputs = self.tokenizer(
                summary,
                truncation=True,
                padding="max_length",
                max_length=128,
            )

            tokenized_inputs["labels"] = is_factual

            return tokenized_inputs

        tokenized_datasets: DatasetDict = self.dataset.map(
            tokenize_function, batched=True
        )

        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Ensure that the WANDB_API_KEY and WANDB_PROJECT are not None or empty
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb_project = os.getenv("WANDB_PROJECT")
        if not wandb_api_key or not wandb_project:
            raise ValueError(
                "WANDB_API_KEY and WANDB_PROJECT must be set in the environment"
            )

        # Training the model with WANDB parameters
        training_args = TrainingArguments(
            report_to="wandb",
            run_name="nli_finetuning_run",  # Optionally, add a run name
            output_dir="./results",
            logging_dir="./logs",
            logging_steps=10,
            do_eval=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=1000,
            eval_steps=25,
        )

        trainer = Trainer(
            model=self.model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=tokenized_datasets["train"],  # training dataset
            eval_dataset=tokenized_datasets["test"],  # evaluation dataset
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        # Evaluate the model
        trainer.evaluate()

        # Save the model
        trainer.save_model("./models/nli_finetuned_model")


if __name__ == "__main__":
    nli = NLI_Finetune()
    nli.finetune()
