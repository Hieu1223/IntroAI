class Model:
    def __init__(self):
        self.dataset
    def train(self):
        raise NotImplementedError("Subclasses need to implement this method")
    def evaluate(self,data):
        raise NotImplementedError("Subclasses need to implement this method")
        