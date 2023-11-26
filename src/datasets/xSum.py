import datasets
import wandb
import torch


class XSum:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        dataset = datasets.load_dataset("EdinburghNLP/xsum")
        train = dataset["train"].shuffle(seed=42).select(range(wandb.config.train_size))
        val = (
            dataset["validation"].shuffle(seed=42).select(range(wandb.config.val_size))
        )

        self.train_tokenized = self.preprocess(train)
        self.val_tokenized = self.preprocess(val)

    def preprocess(self, dataset):
        # extract data as dict
        output_text = []
        skipped = 0
        for i, text in enumerate(dataset):
            document = text["document"]
            tok_sum = text["summary"]

            if wandb.config.use_prompt:
                document = f"Summarize: '{document}' Output:"

            tokenized = self.tokenizer(document)

            # if labels or input_ids are longer than sequence length, skip
            if tokenized.data["input_ids"].shape[1] > wandb.config.sequence_length:
                skipped += 1
                continue

            # convert all to 32bit integers
            entry = {"input_ids": document, "labels": tok_sum}
            output_text.append(entry)

        print(f"Skipped {skipped} examples")
        return output_text
