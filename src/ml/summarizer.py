import src.ml.baseModel as bs
import src.ml.baseModel as bm
import transformers
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM, TaskType
import bitsandbytes as bnb
import wandb

class Summarizer(bs.BaseModel):
    def __init__(self):
        print(f"GPUs available: {torch.cuda.device_count()}")

        # Hyperparameters
        self.model              = wandb.config.model_name
        self.learning_rate      = wandb.config.learning_rate
        self.weight_decay       = wandb.config.weight_decay
        self.epochs             = wandb.config.epochs
        self.batch_size         = wandb.config.batch_size
        self.sequence_length    = wandb.config.sequence_length

        return

        # Tokenizer
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(self.name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Tokenizer: {self.tokenizer.eos_token}; Pad {self.tokenizer.pad_token}")

        # Model
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = "bfloat16",
        )
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none"
        )
        self.model = transformers.LlamaForCausalLM.from_pretrained(self.name, 
            quantization_config = self.bnb_config,
            device_map = "auto"
            )
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def compute_metrics(self, eval_pred):
        pass

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', padding='max_length', max_length=self.sequence_length)

    def collate_fn(self, batch):
        input_ids       = torch.cat([torch.tensor(x['input_ids']) for x in batch])
        attention_mask  = torch.cat([torch.tensor(x['attention_mask']) for x in batch])
        labels          = torch.tensor([x['labels'] for x in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def train(self):
        # Prepare the model for training 
        self.model = prepare_model_for_kbit_training(self.model)

        training_args = transformers.TrainingArguments(
            report_to='wandb',
            output_dir='./results',
            learning_rate=self.lr,
            warmup_ratio= 0.1,
            max_grad_norm= 0.3,
            weight_decay=self.weight_decay,
            load_best_model_at_end=True,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            logging_dir='./logs',
            logging_steps=10,
            do_eval=True,
            metric_for_best_model='eval_loss',
            push_to_hub_model_id="Llama-2-7b-XSum-4bit",
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_steps=50,
            fp16=True,
            gradient_checkpointing=True,        
            optim = "paged_adamw_32bit",
            hub_token="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" # TODO use organization?
        )
        
        trainer = transformers.Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.push_to_hub()

    def predict(self, X):
        pass