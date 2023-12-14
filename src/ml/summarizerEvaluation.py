import src.ml.summarizer as s
import src.datasets.xSum as xSum
import src.utils.utilities as utils
import numpy as np
import torch
import wandb
import evaluate


class LlamarizerEval():
    def __init__(self):
        # Load the model and fetch the important bits
        self.bnb_config = utils.get_bnb_config()
        self.peft_config = utils.get_peft_config()

        wandb_output = utils.load_from_wandb("ernlavr/adv_nlp2023/model_nioagqxs:v0", 
                                             load_in_4bit=False, 
                                             peft_config=None, 
                                             bnb_config=None)
        self.model = wandb_output[0]
        self.tokenizer = wandb_output[1]

        if not self.model.config.pad_token_id:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eval_set = xSum.XSum(self.tokenizer, no_summary=True).val_tokenized

        # Hyperparameters
        self.sequence_length = wandb.config.sequence_length
        self.eval_batch = wandb.config.eval_batch_size

        # Metrics
        self.rouge = evaluate.load("rouge")


    def compute_metrics(self, predictions, labels):
        output = {}
        rouge_output = self.rouge.compute(predictions=predictions, references=labels)

        # Rouge
        output['rouge1'] = rouge_output['rouge1']
        output['rouge2'] = rouge_output['rouge2']
        output['rougeL'] = rouge_output['rougeL']

        wandb.log(output)
        return output


    def push_artifacts_table(self, inputs, preds, labels, metrics):
        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = metrics["rouge1"]
        r2 = metrics["rouge2"]
        rl = metrics["rougeL"]

        text_table = wandb.Table(columns=["inputs", "pred", "ref", "R1", "R2", "RougeL"])

        for (i, p, l) in zip(inputs, preds, labels):
            text_table.add_data(i, p, l, r1, r2, rl)

        wandb.run.log({'Eval_Samples' : text_table})
        

    def run_inference(self, batch, label_length):
        # tokenize the text
        self.model.eval()
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        seq_len = input_ids.shape[1] + label_length

        # generate the summary
        summary_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=seq_len,
            repetition_penalty=1.25,
            length_penalty=-1,
        )

        # decode the summary
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        # remove the input spans from summaries
        for i in range(len(inputs)):
            input_length = len(inputs[i])
            summary[i] = summary[i][input_length:]

        return summary


    def collate_fn(self, batch):
        # get the longest sequence of the batch
        max_length = max([len(x["input_ids"]) for x in batch])
        def padding_fn(x):
            return self.tokenizer.pad(x, max_length=max_length, padding="longest")
        
        def pad_labels(x):
            # dont pad, its ok
            if len(x) == max_length:
                return x

            return np.concatenate((x, [-100] * (max_length - len(x))))

        # pad all the sequences to max_length
        inputs = padding_fn(batch)
        labels = np.array([pad_labels(x) for x in inputs.data['labels']])
        attention_mask = np.array(inputs.data["attention_mask"])
        input_ids = np.array(inputs.data["input_ids"])

        input_ids = torch.tensor(input_ids).to(self.model.device)
        attention_mask = torch.tensor(attention_mask).to(self.model.device)
        labels = torch.tensor(labels).to(self.model.device)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def eval(self):
        self.model.eval()
        
        # evaluate in batches
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=self.eval_batch,
            collate_fn=self.collate_fn,
        )

        # run inference
        for batch in eval_dataloader:
            # Decode labels
            labels = batch["labels"]
            label_length = len(labels[labels != -100])
            labels[labels == -100] = self.tokenizer.pad_token_id
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Run the inference
            preds = self.run_inference(batch, label_length)

            # Compute the metrics
            metrics = self.compute_metrics(preds, labels)

            # Log
            inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            self.push_artifacts_table(inputs, preds, labels, metrics)