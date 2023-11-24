import src.ml.baseModel as bm
import transformers
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM, TaskType
import bitsandbytes as bnb

class NLI(bm.BaseModel):
    def __init__(self, model_name, **kwargs):
        print("Init NLI")

    def compute_metrics(self, eval_pred):
        pass

    def tokenize(self, text):
        pass

    def collate_fn(self, batch):
        pass
    
    def train(self):
        pass

    def evaluate(self):
        print("Evaluating the model")
        