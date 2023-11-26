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
import src.utils.EvalTrainer as et
import evaluate
import numpy as np


from transformers.utils import logging


logging.set_verbosity_debug()
logger = logging.get_logger("transformers")


class Summarizer(bs.BaseModel):
    def __init__(self):
        print(f"GPUs available: {torch.cuda.device_count()}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.model_name = wandb.config.model_name
        self.learning_rate = wandb.config.learning_rate
        self.weight_decay = wandb.config.weight_decay
        self.epochs = wandb.config.epochs
        self.batch_size = wandb.config.batch_size
        self.sequence_length = wandb.config.sequence_length
        self.load_in_4bit = wandb.config.load_in_4bit

        # metrics
        self.rouge = evaluate.load('rouge')

        # Tokenizer
        print("Loading tokenizer")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset
        print("Loading dataset")
        self.dataset = xSum.XSum(self.tokenize)

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
            bias="none",
        )
        self.model = self.get_model()
        if not self.model.config.pad_token_id:
            self.model.config.pad_token_id = self.model.config.eos_token_id
        

    def get_model(self):
        model = None
        if self.load_in_4bit:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=self.bnb_config, device_map="auto"
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, self.peft_config)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name
            )

        return model

    def compute_metrics(self, eval_pred):
        """ Eval_pred consists of a tuple of predictions and labels
            predictions (1, 1, 1024, 50257)
            labels (1, 1, 1024)
        """
        # if eval_pred is torch, make it numpy
        if isinstance(eval_pred[0], torch.Tensor):
            eval_pred = [(x[1].detach().cpu().numpy(), x[1].detach().cpu().numpy()) for x in eval_pred]

        # compute ROUGE
        predictions, labels = np.squeeze(eval_pred[0]), np.squeeze(eval_pred[1])
        # take softmax over logits
        predictions = np.argmax(predictions, axis=-1)

        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        rouge = self.rouge.compute(predictions=predictions, references=labels)
        wandb.log({"rouge1": rouge["rouge1"]})
        wandb.log({"rouge2": rouge["rouge2"]})
        wandb.log({"rougeL": rouge["rougeL"]})
        wandb.log({"rougeLsum": rouge["rougeLsum"]})


        return {'rouge': rouge}

    def tokenize(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
        )

    def collate_fn(self, batch):
        # pad the input_ids and attention_mask to the longest sequence in the batch
        inputs = [x["input_ids"] for x in batch]
        inputs_tokenized = self.tokenizer(inputs, return_tensors='pt', padding='longest')
        inputs_length = inputs_tokenized.data["input_ids"].shape[1]
        
        labels = [x["labels"] for x in batch]
        labels_tokenized = self.tokenizer(labels, return_tensors='pt', padding='max_length', max_length=inputs_length)

        
        input_ids = inputs_tokenized.data["input_ids"]
        attention_mask = inputs_tokenized.data["attention_mask"]

        labels = labels_tokenized.data["input_ids"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def train(self):
        # Prepare the model for training
        training_args = transformers.TrainingArguments(
            report_to="wandb",
            output_dir="./results",
            logging_dir="./logs",
            logging_steps=10,
            do_eval=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=1000,
            eval_steps=25,

            # hyperparameters
            learning_rate=self.learning_rate,
            warmup_ratio=0.1,
            max_grad_norm=0.3,
            weight_decay=self.weight_decay,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=1,
            
            # huggingface
            push_to_hub_model_id="llama2-7bn-" + "4bit-xsum",
            hub_token="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",
            load_best_model_at_end=True,

            # model quantization stuff
            fp16=wandb.config.load_in_4bit,
            gradient_checkpointing=wandb.config.load_in_4bit,        
            optim = "paged_adamw_32bit" if wandb.config.load_in_4bit else "adamw"
        )

        trainer = et.CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset.train_tokenized,
            eval_dataset=self.dataset.val_tokenized,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
        )

        trainer.train()
        trainer.push_to_hub()

    def predict(self, X):
        pass
