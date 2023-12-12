from ctc_score import SummarizationScorer
import numpy as np

class CTC:
    def __init__(self):
        self.scorer = SummarizationScorer(align='D-cnndm')

    def compute(self, docs, preds):
        
        scores = []
        for doc, pred in zip(docs, preds):
            score = self.scorer.score(doc=doc, refs=[], hypo=pred, aspect='consistency')
            scores.append(score)
        return np.mean(scores)