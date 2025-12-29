import csv
from typing import Literal
import numpy as np

class HousingDataset:
    def __init__(self, mode: Literal['train','test']='train'):
        self.y_div_factor = 1000000.0

        with open('dataset/data.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader)

            x, y = [], []
            for row in reader:
                y.append(row[1])
                x.append(row[2:-4])

        def to_float(v):
            try:
                return float(v)
            except ValueError:
                return 0.0

        x = np.array([[to_float(v) for v in row] for row in x], dtype=np.float64)
        y = np.array([float(v) for v in y], dtype=np.float64) / self.y_div_factor

        idx = np.random.permutation(len(x))
        x, y = x[idx], y[idx]

        self.n_features = x.shape[1]
        split_idx = int(0.8 * len(x))

        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        mean = x_train.mean(axis=0)
        std  = x_train.std(axis=0) + 1e-8

        if mode == 'train':
            self.x = (x_train - mean) / std
            self.y = y_train
        else:
            self.x = (x_test - mean) / std
            self.y = y_test


def main():
    dataset = HousingDataset(mode='train')
    print(dataset.n_features)
    print(len(dataset.x))
    print(dataset.x)
    print(dataset.y)
    

if __name__ == "__main__":
    main()
