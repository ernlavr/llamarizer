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
from factCC_module import FactCC
from SummaC_module import SummaC
from ANLI_module import ANLI
from summarization_metrics_module import SummarizationMetrics

class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rouge = eval.load('rouge')
        self.FactCC = FactCC()
        self.SummaC = SummaC()
        self.ANLI = ANLI()
        self.summarization_metrics = SummarizationMetrics()

    def cm(self, preds, labels,sources):
        """ Eval_pred consists of a tuple of predictions and labels
            predictions (1, 1, 1024, 50257)
            labels (1, 1, 1024)
        """
        predictions = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        sources = self.tokenizer.batch_decode(sources, skip_special_tokens=True)
        output_metrics = {}
        summarization_metrics = self.summarization_metrics.compute(sources, predictions)
        rouge = self.rouge.compute(predictions=predictions, references=labels)
        factCC_score = self.FactCC.compute(sources,predictions)
        summaC_score = self.SummaC.compute(sources,predictions)
        anli_score = self.ANLI.compute(sources,predictions)
        output_metrics = output_metrics.update(summarization_metrics)
        output_metrics = output_metrics.update(rouge)   
        output_metrics['factCC'] = factCC_score
        output_metrics['summaC'] = summaC_score
        output_metrics['anli'] = anli_score
        return output_metrics

    def push_artifacts_table(self, epoch, loss, r1, r2, red, nngram1,nngram2,nngram3,comp_score,factCC_score,summaC_score,anli_score ,source, prediction, target):
        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1 = np.mean(r1)
        r2 = np.mean(r2)
        text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2","RED","novel 1gram","novel 2gram","novel 3gram","Compression score", \
                                           "FactCC","SummaC","ANLI","document", "target", "prediction"])

        num_examples = 4
        if prediction.shape[0] < 4:
            num_examples = prediction.shape[0]

        for i in range(num_examples):
            source_i = self.tokenizer.decode(source[i])
            target_i = self.tokenizer.decode(target[i])
            prediction_i = self.tokenizer.decode(prediction[i])

            text_table.add_data(epoch, loss, r1, r2, red, nngram1,nngram2,nngram3,comp_score, factCC_score, summaC_score, anli_score, source_i, target_i, prediction_i)
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
                computed_metrics = self.cm(outputs, inputs["labels"],inputs["input_ids"])
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
        self.push_artifacts_table(self.state.epoch, avg_loss, metrics["rouge1"], metrics["rouge2"], \
            metrics["red_score"],metrics["novel_1gram_ratio"],metrics["novel_2gram_ratio"],metrics["novel_3gram_ratio"],metrics["compression_score"], \
            metrics["FactCC"],metrics["summaC"],metrics["anli"],inputs["input_ids"], outputs, inputs["labels"])
        return {"eval_loss": avg_loss}