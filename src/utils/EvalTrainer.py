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

    def push_artifacts_table(self, epoch, loss, r1, r2, source, prediction, target):
        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = np.mean(r1)
        r2 = np.mean(r2)
        text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2", "document", "target", "prediction"])

        num_examples = 4
        if prediction.shape[0] < 4:
            num_examples = prediction.shape[0]

        for i in range(num_examples):
            source_i = self.tokenizer.decode(source[i])
            target_i = self.tokenizer.decode(target[i])
            prediction_i = self.tokenizer.decode(prediction[i])

            text_table.add_data(epoch, loss, r1, r2, source_i, target_i, prediction_i)
        wandb.run.log({'Training_Samples' : text_table})


    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        metrics = {}

        for inputs in eval_dataloader:
            with torch.no_grad():
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
        wandb.log({"eval_loss": avg_loss})
        self.push_artifacts_table(self.state.epoch, avg_loss, metrics["rouge1"], metrics["rouge2"], inputs["input_ids"], outputs, inputs["labels"])
        return {"eval_loss": avg_loss}