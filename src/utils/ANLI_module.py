#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
# %%
class ANLI:

    def __init__(self,max_length = 512, model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", device = "cuda"):

        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.max_length = max_length
    def compute(self, sources, summaries):
        
        preds = []
        for source, summary in zip(sources, summaries):
            tokenized_input_seq_pair = self.tokenizer.encode_plus(source, summary, max_length=self.max_length,
                                                     return_token_type_ids=True, truncation=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(self.device)

            # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(self.device)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(self.device)
            
            outputs = self.model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=None)
            predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
            preds.append(predicted_probability[0])
        return np.mean(preds)
# %%
