import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AmesHousingDataset:
    def __init__(self, filepath="dataset/ames.csv", mode='train', test_size=0.2, random_state=42):
        self.x = []
        self.y = []
        self.feature_names = []
        self.n_features = 0
        self.y_div_factor = 1e5
        target = "SalePrice"
        # Load CSV
        df = pd.read_csv(filepath)

        # Drop ID columns
        df = df.drop(columns=["Id", "PID"], errors='ignore')

        # Separate target
        y = df[target].values/self.y_div_factor
        df = df.drop(columns=[target])

        # --------------------------
        # Ordinal mappings
        # --------------------------
        qual_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
        bsmt_exposure_mapping = {'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0}
        land_contour_mapping = {'Lvl':3, 'HLS':2, 'Bnk':2, 'Low':1}
        lot_shape_mapping = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
        land_slope_mapping = {'Gtl':3, 'Mod':2, 'Sev':1}
        condition_mapping = {'Norm':3, 'Feedr':2, 'Artery':1,
                             'RRNn':1, 'RRAn':1, 'PosN':2, 'PosA':2}

        ordinal_fields = {
            'Overall Qual': qual_mapping,
            'Overall Cond': qual_mapping,
            'Exter Qual': qual_mapping,
            'Exter Cond': qual_mapping,
            'Bsmt Qual': qual_mapping,
            'Bsmt Cond': qual_mapping,
            'Heating QC': qual_mapping,
            'Kitchen Qual': qual_mapping,
            'Fireplace Qu': qual_mapping,
            'Garage Qual': qual_mapping,
            'Garage Cond': qual_mapping,
            'Pool QC': qual_mapping,
            'Bsmt Exposure': bsmt_exposure_mapping,
            'Land Contour': land_contour_mapping,
            'Lot Shape': lot_shape_mapping,
            'Land Slope': land_slope_mapping,
            'Condition 1': condition_mapping,
            'Condition 2': condition_mapping
        }

        for col, mapping in ordinal_fields.items():
            if col in df.columns:
                df[col] = df[col].fillna('NA').map(mapping).fillna(0)

        # --------------------------
        # Nominal categorical fields to one-hot encode
        # --------------------------
        onehot_columns = [
            'MS SubClass', 'MS Zoning', 'Street', 'Alley', 'Utilities', 'Lot Config',
            'Neighborhood', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
            'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation', 'Heating',
            'Central Air', 'Electrical', 'BsmtFin Type 1', 'BsmtFin Type 2',
            'Garage Type', 'Garage Finish', 'Paved Drive', 'Fence',
            'Misc Feature', 'Sale Type', 'Sale Condition', 'Functional'
        ]

        df[onehot_columns] = df[onehot_columns].fillna('NA')
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_data = encoder.fit_transform(df[onehot_columns])
        onehot_feature_names = encoder.get_feature_names_out(onehot_columns)

        # --------------------------
        # Numeric columns
        # --------------------------
        numeric_cols = [col for col in df.columns if col not in onehot_columns]
        numeric_data = df[numeric_cols].fillna(0).values

        # Combine numeric + one-hot
        X = np.hstack([numeric_data, onehot_data])
        self.feature_names = numeric_cols + list(onehot_feature_names)
        self.n_features = X.shape[1]
        self.feature_names = np.array(self.feature_names)

        
        # --------------------------
        # Train/Test Split
        # --------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if mode == 'train':
            self.x = X_train
            self.y = y_train
        elif mode == 'test':
            self.x = X_test
            self.y = y_test
        else:
            raise ValueError("mode must be 'train' or 'test'")

def main():
    dataset = AmesHousingDataset(mode='train')  # 'train' or 'test'
    
    print("=== Dataset Statistics ===")
    print(f"Number of samples: {dataset.x.shape[0]}")
    print(f"Number of features: {dataset.n_features}")
    print(f"Target (SalePrice) mean: {dataset.y.mean() :.2f}")
    print(f"Target (SalePrice) std: {dataset.y.std() :.2f}")
    print(f"Target (SalePrice) min: {dataset.y.min() :.2f}")
    print(f"Target (SalePrice) max: {dataset.y.max() :.2f}")
    
    print("\n=== Feature Sample ===")
    for i, name in enumerate(dataset.feature_names[:10]):  # Show first 10 features
        print(f"{name}: {dataset.x[0, i]:.3f}")


if __name__ == '__main__':
    main()