import csv
from typing import Literal
from collections import defaultdict
import numpy as np


class HousingDataset:
    


    
    def __init__(self,mode :Literal['train','test'] = 'train', train_split: float = 0.8):
        self.data = []
        self.street2idx = defaultdict(lambda: len(self.street2idx))
        self.city2idx = defaultdict(lambda: len(self.city2idx))
        self.statezip2idx = defaultdict(lambda: len(self.statezip2idx))
        self.country2idx = defaultdict(lambda: len(self.country2idx))
        with open('dataset/data.csv', mode ='r')as file:
            csvFile = csv.reader(file)
            start_idx = 0 if mode == 'train' else int(0.8 *csvFile.line_num)
            end_idx = int(0.8 *csvFile.line_num) if mode == 'train' else csvFile.line_num

            for line in csvFile:
                date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country = line
                street_idx = self.street2idx[street]
                city_idx = self.city2idx[city]
                statezip_idx = self.statezip2idx[statezip]
                country_idx = self.country2idx[country]

                self.data.append([price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view,
                                  condition, sqft_above, sqft_basement, yr_built, yr_renovated,
                                  street_idx, city_idx, statezip_idx, country_idx])
        self.data.pop(0)


def main():
    dataset = HousingDataset(mode='train')
    print(dataset.data[0])
    print(dataset.street2idx)



if __name__ == "__main__":
    main()