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
import src.ml.nli as nli
import evaluate as eval
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        nli_artifact = kwargs.pop("nli_artifact")

        super().__init__(*args, **kwargs)
        self.rouge = eval.load('rouge')
        self.repetition_penalty = wandb.config.repetition_penalty
        self.wandb_num_examples = wandb.config.wandb_num_examples
        if wandb.config.train_with_nli:
            self.nli_model = nli.NLI_Finetune(nli_artifact, device=self.model.device)


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


    def push_artifacts_table(self, epoch, loss, r1, r2, predictions):

        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = np.mean(r1)
        r2 = np.mean(r2)
        text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2", "document", "target", "prediction"])

        num_examples = self.wandb_num_examples
        if len(predictions["document"]) < num_examples:
            num_examples = len(predictions["document"])

        for i in range(num_examples):
            document_i = predictions['document'][i]
            labels_i = predictions['labels'][i]
            prediction_i = predictions['prediction'][i]

            text_table.add_data(epoch, loss, r1, r2, document_i, labels_i, prediction_i)
        wandb.run.log({'Training_Samples' : text_table})

    def decode_example(self, example, skip_special_tokens=False):
        return self.tokenizer.decode(example, skip_special_tokens=skip_special_tokens)
    
    def compute_loss(self, model, inputs, return_outputs=False, evaluation=False):
        """ Overwrite the compute loss method to use the PEFT loss function """
        # If we dont want NLI then just proceed with normal loss!
        if wandb.config.train_with_nli == False:
            return super().compute_loss(model, inputs, return_outputs)


        # Custom loss only for NLI!
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # compute summary loss
        summary_loss = F.nll_loss(
            F.log_softmax(logits, dim=-1).view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean",
            ignore_index=-100,
        )

        # Fetch the prediction span
        preds = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds[labels == -100] = self.tokenizer.eos_token_id

        # Decode source document and predictions
        decoded_inputs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Check NLI on source and predictions
        nli_probs = self.nli_model.infer(decoded_inputs, decoded_preds)
        nli_loss = torch.mean(nli_probs[:, 0]).item()

        # Log
        log_prefix = "train"
        if evaluation:
            log_prefix = "eval"
        wandb.log({f"{log_prefix}/nli_loss": nli_loss})
        wandb.log({f"{log_prefix}/summary_loss": summary_loss.item()})
        
        # Combine losses and return
        loss_final = sum([summary_loss, nli_loss])
        return (loss_final, outputs) if return_outputs else loss_final

    def push_artifacts_table(self, epoch, loss, r1, r2, predictions):

        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = np.mean(r1)
        r2 = np.mean(r2)
        text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2", "document", "target", "prediction"])

        num_examples = self.wandb_num_examples
        if len(predictions["document"]) < num_examples:
            num_examples = len(predictions["document"])

        for i in range(num_examples):
            document_i = predictions['document'][i]
            labels_i = predictions['labels'][i]
            prediction_i = predictions['prediction'][i]

            text_table.add_data(epoch, loss, r1, r2, document_i, labels_i, prediction_i)
        wandb.run.log({'Training_Samples' : text_table})

    def decode_example(self, example, skip_special_tokens=False):
        return self.tokenizer.decode(example, skip_special_tokens=skip_special_tokens)
    

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        result_count = 0
        metrics = {}
        result_summary = {"document" : [], "labels" : [], "prediction" : []}
        

        for inputs in tqdm(eval_dataloader, "Evaluating.."):
            with torch.no_grad():
                # get predictions
                inputs = self._prepare_inputs(inputs)
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True, evaluation=True)
                total_loss += loss.item()
                total_steps += 1

                # Compute metrics and add to dict
                outputs = torch.softmax(outputs.logits, dim=-1)
                outputs = torch.argmax(outputs, dim=-1)

                # replace every outputs index with eos_token_id where label is -100
                outputs[inputs["labels"] == -100] = self.tokenizer.eos_token_id

                # After loss computation, turn the label -100s to pad_token_id
                # This is because we can't decode -100s but we need it for loss computation
                labels = inputs["labels"]
                labels[labels == -100] = self.tokenizer.eos_token_id

                computed_metrics = self.cm(outputs, labels)
                for key, value in computed_metrics.items():
                    if key in metrics:
                        metrics[key].append(value)
                    else:
                        metrics[key] = [value]

                # save the predictions
                for i, _ in enumerate(outputs):
                    if result_count <= 5:
                        result_count += 1
                        result_summary["document"].append(self.decode_example(inputs["input_ids"][i].squeeze()))
                        result_summary["labels"].append(self.decode_example(labels[i].squeeze(), True))
                        result_summary["prediction"].append(self.decode_example(outputs[i].squeeze(), True))

        # log each metrics to wandb
        for key, value in metrics.items():
            val = round(np.mean(value), 4)
            print(f"Logging {key}: {val}")
            wandb.log({key: val})

        # finish up
        avg_loss = total_loss / total_steps
        self.model.train()
        wandb.log({"eval_loss": avg_loss})
        self.push_artifacts_table(self.state.epoch, avg_loss, metrics["rouge1"], metrics["rouge2"], result_summary)
        return {"eval_loss": avg_loss}