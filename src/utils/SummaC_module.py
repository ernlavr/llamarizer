from summac.model_summac import SummaCConv
import torch
import numpy as np

class SummaC:

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device=self.device, start_file="default", agg="mean") 

    def compute(self, sources, summaries):
        scores = self.model.score(sources, summaries)
        print(scores['scores'])
        return np.mean(scores['scores'])