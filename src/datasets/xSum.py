import datasets


class XSum:
    def __init__(self) -> None:
        self.dataset = datasets.load_dataset("EdinburghNLP/xsum")

        # print average sequence length
        print(
            sum([len(x) for x in self.dataset["train"]["document"]])
            / len(self.dataset["train"]["document"])
        )
        self.test_split = self.dataset["test"]
        pass
