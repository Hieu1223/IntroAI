from model import Model
from dataset import HousingDataset
from utils import mse
import numpy as np

class LinearRegression(Model):
    def __init__(self, n_features):
        self.coefficients = np.random.rand(n_features)
        self.n_features = n_features

    def train(self, dataset,steps=1000, lr=0.01):
        for step in range(steps):
            delta_w = np.dot(np.transpose(dataset.x),(np.dot(dataset.x,self.coefficients)-dataset.y))/len(dataset.x) 
            self.coefficients -= lr*delta_w

    def evaluate(self, data):
        return np.dot(data.x, self.coefficients)




def main():
    train_dataset = HousingDataset(mode='train')
    test_dataset = HousingDataset(mode='test')
    model = LinearRegression(n_features=train_dataset.n_features)
    model.train(train_dataset,steps=1000,lr=0.05)
    res = model.evaluate(test_dataset)
    print(mse(res, test_dataset.y))

if __name__ == "__main__":
    main()