import datasets
import wandb
import torch


class XSum:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset("EdinburghNLP/xsum")
        self.train = self.dataset["train"].shuffle(seed=42).select(range(1600))
        self.val = self.dataset["validation"].shuffle(seed=42).select(range(160))

        # hyperparameters
        self.sequence_length = wandb.config.sequence_length

        self.train_tokenized = self.preprocess(self.train)
        self.val_tokenized = self.preprocess(self.val)

    def preprocess(self, dataset):
        # extract data as dict
        output = {"text": [], "summary": []}
        output_text = []
        skipped = 0
        for i, text in enumerate(dataset):
            tok_doc = self.tokenizer(text["document"])
            tok_sum = self.tokenizer(text["summary"])
            tok_doc.data["labels"] = tok_sum["input_ids"]

            # if labels or input_ids are longer than sequence length, skip
            if (
                tok_doc.data["labels"].shape[1] > self.sequence_length
                or tok_doc.data["input_ids"].shape[1] > self.sequence_length
            ):
                skipped += 1
                continue

            # convert all to 32bit integers
            output_text.append(tok_doc)

        print(f"Skipped {skipped} examples")
        return output_text
