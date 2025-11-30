from model import Model


class LinearRegression(Model):
    def __init__(self, n_features):
        self.coefficients = None
        self.n_features = n_features

    def train(self, dataset):
        pass

    def evaluate(self, data):
        return super().evaluate(data)




def main():
    pass



if __name__ == "__main__":
    main()