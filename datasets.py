from dataset import Dataset

class HousingDataset(Dataset):
    def __init__(self, mode,normalize = True,y_div_factor = 100000.0):
        super().__init__(
            y_div_factor= y_div_factor,
            normalize=normalize,
            mode=mode,
            filepath="dataset/1553768847-housing.csv",
            target="median_house_value",
            fill_mean_if_null=['total_bedrooms']
        )

class KCHousingDataset(Dataset):
    def __init__(self, mode,normalize = True,y_div_factor = 100000.0):
        super().__init__(
            y_div_factor = y_div_factor,
            normalize=normalize,
            mode=mode,
            filepath="dataset/kc_final.csv",
            target="price",
            drop_columns=['id','date']
        )


class AmesHousingDataset(Dataset):
    def __init__(self, mode='train', seed=123,normalize = True,onehot = True):
        """
        Subclass of Dataset for Ames Housing CSV
        - Some categorical columns as integer
        - Some as one-hot
        - Others left untouched (numeric or object as-is)
        """
        filepath = "dataset/ames.csv"
        target = "SalePrice"

        drop_columns = ["Id", "PID"]

        drop_if_null = []

        fill_mean_if_null = [
            "Lot Frontage", "Mas Vnr Area", "Garage Yr Blt", "BsmtFin SF 1", "BsmtFin SF 2",
            "Bsmt Unf SF", "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Low Qual Fin SF",
            "Gr Liv Area", "Garage Cars", "Garage Area"
        ]

        # Columns to encode as integer
        parameters = [
            "MS SubClass", "Garage Cars", "Bedroom AbvGr", "Kitchen AbvGr",
            "Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath", "TotRms AbvGrd","Exter Cond",
            "Kitchen Qual", "Functional",
            "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
            "Paved Drive","Roof Style", "Roof Matl",
            "House Style","MS Zoning","Street",
            "Lot Shape", "Land Contour", "Lot Config",
            "Land Slope", "Neighborhood", "Bldg Type",
            "Exterior 1st", "Exterior 2nd",
            "Exter Qual", "Foundation", "Heating", "Heating QC",
            "Central Air", "Electrical", 'Condition 1','Condition 2',
            'Mas Vnr Type','Bsmt Qual','Bsmt Cond', 'Bsmt Exposure','Sale Condition','Sale Type','Fireplace Qu',
            'BsmtFin Type 1','BsmtFin Type 2','Pool QC','Fence','Misc Feature','Alley','Utilities'
        ]

        integer_categorical_columns = [

        ]

        # Columns to one-hot encode
        onehot_columns = [
        ]
        if onehot:
            onehot_columns = parameters
        else:
            integer_categorical_columns = parameters

        super().__init__(
            normalize= normalize,
            filepath=filepath,
            target=target,
            drop_columns=drop_columns,
            drop_if_null=drop_if_null,
            fill_mean_if_null=fill_mean_if_null,
            mode=mode,
            seed=seed,
            y_div_factor=100000.0,
            integer_categorical_columns=integer_categorical_columns,
            onehot_columns=onehot_columns
        )




def main():
    dataset = AmesHousingDataset(mode='train')
    print("Number of features:", dataset.n_features)
    print("Feature names:", dataset.feature_names)
    print("Number of samples:", len(dataset.x))


if __name__ == '__main__':
    main()
