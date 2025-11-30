class Model:
    def __init__(self, dataset):
        self.dataset = dataset
    def train(self):
        raise NotImplementedError("Subclasses need to implement this method")
    def evaluate(self,data):
        raise NotImplementedError("Subclasses need to implement this method")
        