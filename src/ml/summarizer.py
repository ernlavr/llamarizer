import evaluate
import numpy as np
import torch
import transformers
import wandb
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import transformers
import src.ml.baseModel as bs
import src.datasets.xSum as xSum
import src.utils.EvalTrainer as et
import src.utils.logging as logUtils
import evaluate
import numpy as np
import os


from transformers.utils import logging

import src.datasets.xSum as xSum
import src.ml.baseModel as bs
import src.utils.EvalTrainer as et

logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class Summarizer(bs.BaseModel):
    def __init__(self, run):
        print(f"GPUs available: {torch.cuda.device_count()}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.model_name = wandb.config.model_name
        self.learning_rate = wandb.config.learning_rate
        self.weight_decay = wandb.config.weight_decay
        self.epochs = wandb.config.epochs
        self.train_batch_size = wandb.config.batch_size
        self.eval_batch_size = wandb.config.eval_batch_size
        self.eval_steps = wandb.config.eval_steps
        self.sequence_length = wandb.config.sequence_length
        self.load_in_4bit = wandb.config.load_in_4bit
        self.wandb_run = run

        # metrics
        self.rouge = evaluate.load("rouge")

        # Tokenizer
        print("Loading tokenizer")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            #self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        # Dataset
        print("Loading dataset")
        self.dataset = xSum.XSum(self.tokenizer)
         # 0.5 epoch
        self.warm_up_steps = int(len(self.dataset.train_tokenized) / self.batch_size / 2)

        # Model
        print("Loading model")
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
        )
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none"
        )
        self.model = self.get_model()
        if not self.model.config.pad_token_id:
            self.model.config.pad_token_id = self.model.config.eos_token_id
            #self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        

    def get_model(self):
        model = None
        if self.load_in_4bit:
            # TODO: review model loading according to llama-recipes
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                quantization_config=self.bnb_config,
                device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, self.peft_config)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)

        return model

    def compute_metrics(self, eval_pred):
        """Eval_pred consists of a tuple of predictions and labels
        predictions (1, 1, 1024, 50257)
        labels (1, 1, 1024)
        """
        # if eval_pred is torch, make it numpy
        if isinstance(eval_pred[0], torch.Tensor):
            eval_pred = [
                (x[1].detach().cpu().numpy(), x[1].detach().cpu().numpy())
                for x in eval_pred
            ]

        # compute ROUGE
        input_ids, predictions, labels = np.squeeze(eval_pred[0]), np.squeeze(eval_pred[1]), np.squeeze(eval_pred[2])
        
        # take softmax over logits
        predictions = np.argmax(predictions, axis=-1)

        # Switch -100 to pad_token_id as an easy hack to decode
        labels[labels == -100] = self.tokenizer.pad_token_id
    
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge = self.rouge.compute(predictions=predictions, references=labels)
        wandb.log({"rouge1": rouge["rouge1"]})
        wandb.log({"rouge2": rouge["rouge2"]})
        wandb.log({"rougeL": rouge["rougeL"]})
        wandb.log({"rougeLsum": rouge["rougeLsum"]})

        print(rouge)
        logUtils.push_artifacts_table("n/a", "n/a", rouge["rouge1"], rouge["rouge2"], )

        return {'rouge': rouge}

    def tokenize(self, text):
        return self.tokenizer(
            text,
            padding="longest",
            max_length=self.sequence_length,
            truncation=True,
        )

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

    def save_model(self, trainer):
        # save the checkpoints of best performing model
        run_id = wandb.run.id
        trainer.save_model(f"./results/best_model_{run_id}")

        # upload the best model weights to wandb
        artifact = wandb.Artifact(f'model_{run_id}', type='model')
        artifact.add_dir(f"./results/best_model_{run_id}")
        wandb.run.log_artifact(artifact)
        artifact.wait() # wait for artifact to finish uploading

    def train(self):
        # Prepare the model for training
        training_args = transformers.TrainingArguments(
            # logging
            report_to="wandb",
            output_dir="./results",
            logging_dir="./logs",
            logging_steps=10,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=self.eval_steps,
            dataloader_pin_memory=False,
            include_inputs_for_metrics=True,

            # hyperparameters
            learning_rate=self.learning_rate,
            warmup_ratio=0.1,
            max_grad_norm=0.3,
            weight_decay=self.weight_decay,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            warmup_steps=self.warm_up_steps,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=8,
            
            # huggingface
            push_to_hub_model_id="llama2-7bn-" + "4bit-xsum",
            hub_token="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",
            load_best_model_at_end=True,
            # model quantization stuff
            fp16=wandb.config.load_in_4bit,
            gradient_checkpointing=wandb.config.load_in_4bit,   
            gradient_accumulation_steps=1,     
            optim = "paged_adamw_32bit" if wandb.config.load_in_4bit else "adamw_torch"
        )
        
        trainer = et.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset.train_tokenized,
            eval_dataset=self.dataset.val_tokenized,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        
        if wandb.config.save_model_at_end:
            self.save_model(trainer)

    def predict(self, X):
        pass
