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
        self.repetition_penalty = wandb.config.repetition_penalty

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
        
        num_examples = 4
        if prediction.shape[0] < 4:
            num_examples = prediction.shape[0]

        for i in range(num_examples):
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

        # TODO: Experiment with no min_length and pad predictions, or no max_length and pad labels!
        for inputs in eval_dataloader:
            with torch.no_grad():
                # loop over each example in batch inputs["input_ids"] one by one and generate loss
                outputs = torch.zeros(inputs["input_ids"].shape[0], inputs["input_ids"].shape[1], dtype=torch.int).to(self.model.device)

                # Generate and process examples one by one to save memory ;_; (peasant solution)
                for i in range(inputs["input_ids"].shape[0]):
                    # get the output length
                    output_length = inputs["labels"].shape[1] + inputs["input_ids"].shape[1]
                    inputs_i = {key: value[i].unsqueeze(0) for key, value in inputs.items()}

                    inputs_i = self._prepare_inputs(inputs_i)
                    predictions = self.model.generate(**inputs_i, 
                                                    repetition_penalty=self.repetition_penalty, 
                                                    max_length=output_length, # length output equal to target
                                                    min_length=output_length, # to avoid padding issues
                                                    output_scores=True, 
                                                    return_dict_in_generate=True)
                    
                    # Format the output. Convert to a tensor, flip to BxSxV, and argmax
                    predictions = torch.stack(predictions.scores)
                    predictions = predictions.permute(1, 0, 2)
                    predictions = torch.softmax(predictions, dim=-1)
                    predictions = predictions.view(-1, self.model.config.vocab_size)
                    labels = inputs_i["labels"].view(-1)

                    # Compute loss 
                    loss = self.cross_entropy_loss(predictions, labels, self.tokenizer.pad_token_id)
                    total_loss += loss.item()
                    total_steps += 1

                    # as a last step, replace inputs["input_ids"] with the generated output. Very careful with bugs here!!
                    outputs[i] = torch.argmax(predictions, dim=-1)

                # Compute metrics for the whole batch
                computed_metrics = self.cm(inputs["input_ids"], inputs["labels"])
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