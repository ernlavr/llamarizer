import src.ml.summarizer as s
import src.datasets.xSum as xSum
import torch
import wandb
import evaluate


class LlamarizerEval():
    def __init__(self):
        # Load the model and fetch the important bits
        llamarizer = s.Summarizer()
        self.model = llamarizer.model
        self.tokenizer = llamarizer.tokenizer
        self.eval_set = xSum.XSum(self.tokenizer, no_summary=True).val_tokenized
        self.collate_fn = llamarizer.collate_fn

        # Hyperparameters
        self.sequence_length = llamarizer.sequence_length
        self.eval_batch = llamarizer.eval_batch_size

        # Metrics
        self.rouge = evaluate.load("rouge")


    def compute_metrics(self, predictions, labels):
        output = {}
        rouge_output = self.rouge.compute(predictions=predictions, references=labels)


        # Rouge
        output['rouge1'] = rouge_output['rouge1']
        output['rouge2'] = rouge_output['rouge2']
        output['rougeL'] = rouge_output['rougeL']

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
        

    def run_inference(self, batch):
        # tokenize the text
        self.model.eval()
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)

        # generate the summary
        summary_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.sequence_length,
            repetition_penalty=1.25,
        )

        # decode the summary
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)

        # remove the input spans from summaries
        for i in range(len(inputs)):
            input_length = len(inputs[i])
            summary[i] = summary[i][input_length:]

        return summary


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
            labels[labels == -100] = self.tokenizer.pad_token_id
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Run the inference
            preds = self.run_inference(batch)

            # Compute the metrics
            metrics = self.compute_metrics(preds, labels)

            # Log
            inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            self.push_artifacts_table(inputs, preds, labels, metrics)