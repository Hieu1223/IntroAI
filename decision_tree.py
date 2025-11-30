from model import Model


class DecisionTree(Model):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
    def train(self,dataset):
        pass
    def evaluate(self, data):
        return super().evaluate(data)
    



def main():
    pass



if __name__ == "__main__":
    main()