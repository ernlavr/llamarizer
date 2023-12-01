import datasets
import wandb
import torch
import numpy as np


class XSum:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        dataset = datasets.load_dataset("EdinburghNLP/xsum")
        train = dataset["train"].shuffle(seed=42).select(range(wandb.config.train_size))
        val = (
            dataset["validation"].shuffle(seed=42).select(range(wandb.config.val_size))
        )

        # self.train_tokenized = self.preprocess(train)
        # self.val_tokenized = self.preprocess(val)
        self.prompt = (f"Summarize this dialog: '{{dialog}}'; Summary:")
        self.skipped_counter = 0
        train_set = train.map(self.apply_prompt_template)
        self.train_tokenized = self.preprocess(train_set)

        val_set = val.map(self.apply_prompt_template)
        self.val_tokenized = self.preprocess(val_set)

        

    def remove_empties(self, dataset):
        """ Some examples are empty after preprocessing. Remove them. """
        for i in dataset:
            if len(self.tokenizer.encode(i["document"])) > wandb.config.sequence_length:
                dataset.remove(i)

    def apply_prompt_template(self, example):
        return {
            "document": self.prompt.format(dialog=example["document"]),
            "summary": example["summary"],
        }

    def tokenize_add_label(self, examples):
        tokenizer = self.tokenizer
        prompt = tokenizer.encode(tokenizer.bos_token + examples["document"], add_special_tokens=False)
        summary = tokenizer.encode(examples["summary"] + tokenizer.eos_token, add_special_tokens=False)

        # Skip examples that are too long. This will create empty examples that need to be removed later
        if len(prompt) > wandb.config.sequence_length:
            self.skipped_counter += 1
            return
    
        sample = {
            "input_ids": np.array(prompt + summary),
            "attention_mask": np.array([1] * (len(prompt) + len(summary))),
            "labels": np.array([-100] * len(prompt) + summary)
        }
        return sample

    def preprocess(self, dataset):
        tokenizer = self.tokenizer
        output = []
        for examples in dataset:
            prompt = tokenizer.encode(tokenizer.bos_token + examples["document"], add_special_tokens=False)
            summary = tokenizer.encode(examples["summary"] + tokenizer.eos_token, add_special_tokens=False)

            # Skip examples that are too long. This will create empty examples that need to be removed later
            if len(prompt) > wandb.config.sequence_length:
                self.skipped_counter += 1
                continue
        
            sample = {
                "input_ids": np.array(prompt + summary),
                "attention_mask": np.array([1] * (len(prompt) + len(summary))),
                "labels": np.array([-100] * len(prompt) + summary)
            }
            output.append(sample)
        print(f"Skipped {self.skipped_counter} examples")
        return output
