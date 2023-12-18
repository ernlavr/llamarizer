import src.ml.summarizer as s
import src.datasets.xSum as xSum
import src.utils.utilities as utils
from src.utils.SummarizationMetrics import FactCC, ANLI, SummaC, SummarizationMetrics
from src.utils.BARTScore.bart_score import BARTScorer
import nltk
import numpy as np
import torch
import wandb
import evaluate


class LlamarizerEval():
    def __init__(self):
        # Load the model and fetch the important bits
        self.bnb_config = utils.get_bnb_config()
        self.peft_config = utils.get_peft_config()

        print(wandb.config.model_name)

        wandb_output = utils.load_from_wandb(wandb.config.model_name, 
                                             load_in_4bit=wandb.config.load_in_4bit, 
                                             peft_config=self.peft_config, 
                                             bnb_config=self.bnb_config)
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

        # Download the nltk punkt tokenizer
        nltk.download('punkt')

        # Metrics
        self.rouge = evaluate.load("rouge")
        self.factcc = FactCC('cuda:1')
        self.anli = ANLI('cuda:1')
        #self.summac = SummaC('cuda:1')
        #self.bart_scorer = BARTScorer(checkpoint='facebook/bart-large-cnn')
        self.summarization_metrics = SummarizationMetrics()
        print("Initialized LlamarizerEval")


    def compute_metrics(self, inputs, predictions, labels):
        print("Computing metrics")
        output = {}
        rouge_output = self.rouge.compute(predictions=predictions, references=labels,use_aggregator=False)
        summary_metrics_out = self.summarization_metrics.compute(inputs, predictions)

        # # Compute FactCC
        FactCC_score = self.factcc.compute(inputs, predictions)

        # # Compute ANLI
        ANLI_score = self.anli.compute(inputs, predictions)

        # # Compute SummaC
        # SummaC_score = self.summac.compute(inputs, predictions)

        # # Compute BART
        # BARTscores = self.bart_scorer.score(labels, predictions, batch_size=self.eval_batch)
        
        # Rouge
        output.update(summary_metrics_out)
        output['rouge1'] = rouge_output['rouge1']
        output['rouge2'] = rouge_output['rouge2']
        output['rougeL'] = rouge_output['rougeL']
        output['FactCC'] = FactCC_score
        output['ANLI'] = ANLI_score
        output['SummaC'] = [-1]  * len(rouge_output['rouge1']) #SummaC_score
        output['BARTScore'] = [-1]  * len(rouge_output['rouge1']) #BARTscores

        wandb.log(output)
        return output


    def push_artifacts_table(self, inputs, preds, labels, metrics):
        """ Returns a wandb.Table object containing all the artifacts
            in the run
        """
        r1s = metrics["rouge1"]
        r2s = metrics["rouge2"]
        rls = metrics["rougeL"]
        red_score=metrics["red_score"]
        novel_1gram_ratio=metrics["novel_1gram_ratio"]
        novel_2gram_ratio=metrics["novel_2gram_ratio"]
        novel_3gram_ratio=metrics["novel_3gram_ratio"]
        compression_score=metrics["compression_score"]
        FactCC_scores=metrics["FactCC"]
        ANLI_scores=metrics["ANLI"]
        SummaC_scores=metrics["SummaC"]
        BARTScores=metrics["BARTScore"]

        # print all metric types
        print(f"r1s: {type(r1s)}")
        print(f"r2s: {type(r2s)}")
        print(f"rls: {type(rls)}")
        print(f"red_score: {type(red_score)}")
        print(f"novel_1gram_ratio: {type(novel_1gram_ratio)}")
        print(f"novel_2gram_ratio: {type(novel_2gram_ratio)}")
        print(f"novel_3gram_ratio: {type(novel_3gram_ratio)}")
        print(f"compression_score: {type(compression_score)}")
        print(f"FactCC_scores: {type(FactCC_scores)}")
        print(f"ANLI_scores: {type(ANLI_scores)}")
        print(f"SummaC_scores: {type(SummaC_scores)}")
        print(f"BARTScores: {type(BARTScores)}")

        text_table = wandb.Table(columns=["inputs", "pred", "ref", "R1", "R2", "RougeL",
                                          'red_score', 'novel_1gram_ratio', 'novel_2gram_ratio', 'novel_3gram_ratio', 'compression_score', 'FactCC', 'ANLI', 'SummaC', 'BARTScore'])

        for (i, p, l,r1,r2,rl,rs,novel_1g,novel_2g,novel_3g,compression_sc,FactCC_score,ANLI_score,SummaC_score,BARTScore ) in zip(inputs, preds, labels,r1s,r2s,rls,red_score,novel_1gram_ratio,novel_2gram_ratio,novel_3gram_ratio,compression_score,FactCC_scores,ANLI_scores,SummaC_scores,BARTScores):
            text_table.add_data(i, p, l, r1,r2,rl,rs,novel_1g,novel_2g,novel_3g,compression_sc,FactCC_score,ANLI_score,SummaC_score,BARTScore)

        wandb.run.log({'Eval_Samples' : text_table})
        

    def run_inference(self, batch, label_length):
        # tokenize the text
        self.model.eval()
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        seq_len = input_ids.shape[1] + label_length

        # generate the summary
        # print devices
        print(f"input_ids device: {input_ids.device}")
        print(f"attention_mask device: {attention_mask.device}")
        print(f"model device: {self.model.device}")
        
        input_ids.to(self.model.device)
        attention_mask.to(self.model.device)

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

    def merge_results(self, buffer, metrics):
        """ Merges the results of the metrics into a single dictionary """
        for key in metrics.keys():
            if key not in buffer:
                buffer[key] = []
            buffer[key].append(metrics[key])

        return buffer

    def eval(self):
        print("Starting evaluation")
        self.model.eval()
        
        # evaluate in batches
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_set,
            batch_size=self.eval_batch,
            collate_fn=self.collate_fn,
        )

        # run inference
        metric_accumulation = {}
        for batch in eval_dataloader:
            # Decode labels
            labels = batch["labels"]
            label_length = len(labels[labels != -100])
            labels[labels == -100] = self.tokenizer.pad_token_id
            labels.to(self.model.device)
            labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Run the inference
            preds = self.run_inference(batch, label_length)
            inputs = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

            # Compute the metrics
            metrics = self.compute_metrics(inputs, preds, labels)
            metric_accumulation = self.merge_results(metric_accumulation, metrics)

            # Log
            self.push_artifacts_table(inputs, preds, labels, metrics)

        # Log the mean + std of each metrics
        for key in metric_accumulation.keys():
            # compute
            mean = np.mean(metric_accumulation[key])
            std = np.std(metric_accumulation[key])

            # round and print
            print(f"{key}: {round(mean, 3)} +- {round(std, 3)}")
