import datasets


class XSum:
    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("EdinburghNLP/xsum")
        self.train = self.dataset["train"].shuffle(seed=42).select(range(1000))
        self.val = self.dataset["validation"].shuffle(seed=42).select(range(100))