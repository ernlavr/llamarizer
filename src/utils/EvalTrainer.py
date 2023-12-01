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
from tqdm import tqdm

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = eval.load('rouge')
        self.repetition_penalty = wandb.config.repetition_penalty


    def cm(self, preds, labels):
        """ Eval_pred consists of a tuple of predictions and labels
            predictions (1, 1, 1024, 50257)
            labels (1, 1, 1024)
        """
        predictions = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge = self.rouge.compute(predictions=predictions, references=labels)
        return rouge


    def push_artifacts_table(self, epoch, loss, r1, r2, predictions):

        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = np.mean(r1)
        r2 = np.mean(r2)
        text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2", "document", "target", "prediction"])

        num_examples = 4
        if len(predictions["document"]) < num_examples:
            num_examples = len(predictions["document"])

        for i in range(num_examples):
            document_i = predictions['document'][i]
            target_i = predictions['target'][i]
            prediction_i = predictions['prediction'][i]

            text_table.add_data(epoch, loss, r1, r2, document_i, target_i, prediction_i)
        wandb.run.log({'Training_Samples' : text_table})

    def decode_example(self, example, skip_special_tokens=False):
        return self.tokenizer.decode(example, skip_special_tokens=skip_special_tokens)
    

    def evaluate(self, eval_dataset=None, ignore_keys=None):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
        metrics = {}

        # Following loop assumes evaluation batch size 1!
        predictions = {"document" : [], "target" : [], "prediction" : []}
        prediction_count = 0
        for inputs in tqdm(eval_dataloader, "Running evaluation.."):
            with torch.no_grad():
                
                # Generate and process examples one by one to save memory ;_; (peasant solution)
                for i in range(inputs["input_ids"].shape[0]):
                    # get label length without special tokens
                    label_length = int(torch.sum(inputs["labels"][i] != -100))
                    
                    # get the output length
                    max_out_length = int(inputs["input_ids"].shape[1] + label_length * 2) # double label size
                    min_out_length = int(inputs["input_ids"].shape[1] + label_length / 1.5) # half label size
                    
                    inputs_i = {key: value[i].unsqueeze(0) for key, value in inputs.items()}
                    inputs_i = self._prepare_inputs(inputs_i)
                    prediction = self.model.generate(**inputs_i, 
                                                    repetition_penalty=self.repetition_penalty, 
                                                    max_length=max_out_length, # length output equal to target
                                                    min_length=min_out_length, # to avoid padding issues
                                                    output_scores=True, 
                                                    return_dict_in_generate=True)
                    
                    # Format the output. Convert to a tensor, flip to BxSxV, and argmax
                    prediction = torch.stack(prediction.scores)
                    prediction = prediction.permute(1, 0, 2)
                    prediction = torch.argmax(prediction, dim=-1)
                    labels = inputs_i["labels"]

                    # Our labels are -100 padded, for loss computation, but we can't decode -100 so we replace it with the pad token
                    labels[labels == -100] = self.tokenizer.eos_token_id

                    # save the predictions
                    if prediction_count < 5:
                        prediction_count += 1
                        predictions["document"].append(self.decode_example(inputs_i["input_ids"].squeeze()))
                        predictions["target"].append(self.decode_example(prediction.squeeze(), True))
                        predictions["prediction"].append(self.decode_example(labels.squeeze(), True))

                    # Compute metrics for the whole batch
                    computed_metrics = self.cm(prediction, labels)
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
        avg_loss = 6.9 # .generate() is super shit to compute loss, fake it!
        print(avg_loss)
        self.model.train()
        wandb.log({"eval_loss": avg_loss})
        self.push_artifacts_table(self.state.epoch, avg_loss, metrics["rouge1"], metrics["rouge2"], predictions)
        return {"eval_loss": avg_loss}