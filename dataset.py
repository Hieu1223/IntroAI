import csv
from typing import Literal
from collections import defaultdict
import numpy as np

class HousingDataset:
    def __init__(self, mode: Literal['train','test']='train'):
        self.x = []
        self.y = []
        self.feature_names = {}  # dictionary to map column index -> feature name
        self.n_features = 0
        self.y_div_factor = 100000.0  # to scale down house prices
        intify_buffer = defaultdict(dict) 
        next_int = defaultdict(int)       

        # -------------------------------
        # 1. Read CSV and convert to numeric
        # -------------------------------
        with open(f'dataset/train.csv', 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            self.feature_names = {i-1: name for i, name in enumerate(header[1:-1])}  # map index -> name
            self.n_features = len(header) - 2  # exclude Id and SalePrice

            x = []
            y = []
            for row in reader:
                new_row = []
                for i, item in enumerate(row):
                    if i == 0:  # skip Id
                        continue
                    try:
                        new_row.append(float(item))
                    except ValueError:
                        if item not in intify_buffer[i]:
                            intify_buffer[i][item] = next_int[i]
                            next_int[i] += 1
                        new_row.append(intify_buffer[i][item])
                y.append(new_row[-1]/self.y_div_factor)      # last column = SalePrice
                x.append(new_row[:-1])     # all other columns = features

        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)

        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)
        x_norm = (x - self.x_min) / (self.x_max - self.x_min + 1e-8)  # avoid divide by zero

        split_idx = int(0.8 * len(x_norm))
        if mode == 'train':
            self.x = x_norm[:split_idx]
            self.y = y[:split_idx]
        else:
            self.x = x_norm[split_idx:]
            self.y = y[split_idx:]

def main():
    dataset = HousingDataset(mode='train')
    print(len(dataset.x))
    print(np.max(dataset.y), np.min(dataset.y))  

if __name__ == "__main__":
    main()