#%%
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
import evaluate as eval
import numpy as np
import logging
logging.disable(logging.WARNING)

class FactCC:
    def __init__(self, model_path='manueldeprada/FactCC'):
        """
        Initializes the FactCC class.

        Args:
            model_path (str): The path or name of the pre-trained model to be used. Defaults to 'manueldeprada/FactCC'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    def compute(self, references,predictions):
            """
            Computes the FactCC score for a list of predicted sentences and their corresponding reference sentences.

            Args:
                predictions (list): A list of predicted sentences.
                references (list): A list of reference sentences.

            Returns:
                float: The FactCC score.
            """
            to_eval = list(zip(predictions, references))
            predictions = []
            for pred, ref in to_eval:
                input_dict = self.tokenizer(pred, ref, max_length=512, padding='max_length', truncation='longest_first', return_tensors='pt')
                logits = self.model(**input_dict).logits
                softmax_probs = torch.nn.functional.softmax(logits, dim=1)
                predictions.append(softmax_probs[0][0].item())
                     
            FactCC_score = np.mean(predictions)
            return FactCC_score

# %%
