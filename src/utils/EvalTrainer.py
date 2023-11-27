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
import torch.nn.functional as F

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = eval.load('rouge')
        self.generation_config = self.get_gen_config()

    def get_gen_config(self):
        config = transformers.GenerationConfig(
            repetition_penalty=1.25,
            pad_token_id=self.tokenizer.eos_token_id,
            
            # Output variables
            output_scores=True,
            return_dict_in_generate=True,
        )
        return config

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
        for i in range(2):
            source_i = self.tokenizer.decode(source[i])
            target_i = self.tokenizer.decode(target[i])
            prediction_i = self.tokenizer.decode(prediction[i])

            text_table.add_data(epoch, loss, r1, r2, source_i, target_i, prediction_i)
        wandb.run.log({'Training_Samples' : text_table})

    def compute_loss(self, model, inputs, evaluation=False):
        """
        Compute cross-entropy loss using PyTorch.

        Parameters:
        - predictions: Predicted probabilities (PyTorch tensor)
        - targets: True labels (PyTorch tensor, one-hot encoded)

        Returns:
        - Cross-entropy loss
        """
        loss = super().compute_loss(model, inputs)
        return loss
    
    def cross_entropy_loss(self, predictions, targets, pad_token_id):
        """ Compute cross-entropy loss using PyTorch.
            predictions: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
        """
        # compute loss
        loss = F.cross_entropy(predictions, targets, ignore_index=pad_token_id)
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        metrics = {}

        for inputs in eval_dataloader:
            with torch.no_grad():
                # get predictions
                output_length = inputs["labels"].shape[1] + inputs["input_ids"].shape[1]
                inputs = self._prepare_inputs(inputs)
                outputs = self.model.generate(**inputs, 
                                              repetition_penalty=1.25, 
                                              max_length=output_length, 
                                              min_length=output_length,
                                              output_scores=True, 
                                              return_dict_in_generate=True)
                
                # TODO: Experiment with no min_length and pad predictions, or no max_length and pad labels!
                
                # Format the output. Convert to a tensor, flip to BxSxV, and argmax
                outputs = torch.stack(outputs.scores)
                outputs = outputs.permute(1, 0, 2)
                outputs = torch.softmax(outputs, dim=-1)

                # Compute loss 
                loss = self.cross_entropy_loss(outputs.view(-1, self.model.config.vocab_size), inputs["labels"].view(-1), self.tokenizer.pad_token_id)
                total_loss += loss.item()
                total_steps += 1

                # Compute metrics and add to dict
                outputs = torch.argmax(outputs, dim=-1)
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
        print(avg_loss)
        self.model.train()
        wandb.log({"eval_loss": avg_loss})
        self.push_artifacts_table(self.state.epoch, avg_loss, metrics["rouge1"], metrics["rouge2"], inputs["input_ids"], outputs, inputs["labels"])
        return {"eval_loss": avg_loss}