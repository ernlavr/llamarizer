import torch
import wandb
from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import transformers
import src.ml.baseModel as bs
import src.datasets.xSum as xSum
import evaluate as eval
import numpy as np

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = eval.load('rouge')

    def cm(self, preds, labels):
        """ Eval_pred consists of a tuple of predictions and labels
            predictions (1, 1, 1024, 50257)
            labels (1, 1, 1024)
        """
        predictions = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge = self.rouge.compute(predictions=predictions, references=labels)
        return rouge

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        metrics = {}

        for inputs in eval_dataloader:
            # get predictions
            inputs = self._prepare_inputs(inputs)
            loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
            total_loss += loss.item()
            total_steps += 1

            # Compute metrics and add to dict
            outputs = torch.argmax(outputs.logits, dim=-1)
            computed_metrics = self.cm(outputs, inputs["labels"])
            for key, value in computed_metrics.items():
                if key in metrics:
                    metrics[key].append(value)
                else:
                    metrics[key] = [value]

        # log each metrics to wandb
        for key, value in metrics.items():
            val = round(np.mean(value), 4)
            print(f"Logging {key}: {val}")
            wandb.log({key: val})

        # finish up
        avg_loss = total_loss / total_steps
        self.model.train()
        return {"eval_loss": avg_loss}