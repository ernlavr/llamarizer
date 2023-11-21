import abc

class BaseModel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def score(self, X, y):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass