import src.ml.baseModel as bm


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
