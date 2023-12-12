import transformers
import torch.nn as nn
import torch

class NliTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        # Fetch the class weights and then proceed with the init
        self.class_weights = kwargs.pop("class_weights")
        super().__init__(*args, **kwargs)
        

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        cw = torch.tensor(self.class_weights, device=model.device, dtype=torch.float32)
        loss_fct = nn.CrossEntropyLoss(weight=cw)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss