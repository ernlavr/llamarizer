import abc


class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def collate_fn(self, batch):
        pass

    @abc.abstractmethod
    def tokenize(self, text):
        pass
