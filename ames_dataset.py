import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AmesHousingDataset:
    def __init__(self, filepath="dataset/ames.csv", mode='train', test_size=0.2, random_state=42):
        self.y_div_factor = 1e5
        
        df = pd.read_csv(filepath)

        drop_cols = [
            "Id", "PID", 'Pool QC', 'Misc Feature', 'Alley', 'Fence',
            'Condition 2', 'Utilities', 'Bsmt Exposure', 'Heating', 'Garage Qual','Neighborhood'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        target = "SalePrice"
        y = df[target].values / self.y_div_factor
        df = df.drop(columns=[target])

        numerical_features = [
            'Lot Frontage', 'Garage Yr Blt', 'Mas Vnr Area', 'Bsmt Half Bath', 
            'Bsmt Full Bath', 'BsmtFin SF 1', 'Total Bsmt SF', 'Bsmt Unf SF', 
            'BsmtFin SF 2', 'Garage Area'
        ]
        categorical_features = [
            'Mas Vnr Type', 'Fireplace Qu', 'Garage Finish', 'Garage Type', 
            'Bsmt Qual', 'BsmtFin Type 1', 'Bsmt Cond', 'Functional',
            'Electrical', 'Garage Cond', 'Roof Matl', 'Land Contour', 'Garage Cars'
        ]

        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median()) 
        for feature in categorical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].mode()[0])

        qual_mapping = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0}
        lot_shape_mapping = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
        land_slope_mapping = {'Gtl':3, 'Mod':2, 'Sev':1}
        condition_mapping = {'Norm':3, 'Feedr':2, 'Artery':1, 'RRNn':1, 'RRAn':1, 'PosN':2, 'PosA':2}
        land_contour_mapping = {'Lvl':3, 'HLS':2, 'Bnk':2, 'Low':1}

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
            'Lot Shape': lot_shape_mapping,
            'Land Slope': land_slope_mapping,
            'Condition 1': condition_mapping,
            'Land Contour': land_contour_mapping 
        }

        for col, mapping in ordinal_fields.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        onehot_columns = [
            'MS SubClass', 'MS Zoning', 'Street', 'Lot Config',
            'Bldg Type', 'House Style', 'Roof Style',
            'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation',
            'Central Air', 'BsmtFin Type 1', 'BsmtFin Type 2',
            'Garage Type', 'Garage Finish', 'Paved Drive', 'Sale Type', 'Sale Condition'
        ]

        df[onehot_columns] = df[onehot_columns].fillna('NA')
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_data = encoder.fit_transform(df[onehot_columns])
        onehot_feature_names = encoder.get_feature_names_out(onehot_columns)

        numeric_df = df.drop(columns=onehot_columns)
        
        numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        X = np.hstack([numeric_df.values, onehot_data])
        self.feature_names = np.array(list(numeric_df.columns) + list(onehot_feature_names))
        self.n_features = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        self.x = scaler.fit_transform(X_train) if mode == 'train' else scaler.fit_transform(X_train)

        if mode == 'test':
            scaler.fit(X_train)
            self.x = scaler.transform(X_test)
            self.y = y_test
        else:
            self.x = scaler.fit_transform(X_train)
            self.y = y_train
def main():
    dataset = AmesHousingDataset(mode='train')
    
    print("=== Dataset Statistics ===")
    print(f"Number of samples: {dataset.x.shape[0]}")
    print(f"Number of features: {dataset.n_features}")
    print(f"Target (SalePrice) mean: {dataset.y.mean():.2f}")
    print(f"Target (SalePrice) std: {dataset.y.std():.2f}")
    print(f"Target (SalePrice) min: {dataset.y.min():.2f}")
    print(f"Target (SalePrice) max: {dataset.y.max():.2f}\n")

    num_cols = pd.DataFrame(dataset.x, columns=dataset.feature_names).select_dtypes(include=[np.number]).columns
    num_summary = pd.DataFrame(dataset.x, columns=dataset.feature_names)[num_cols].describe().T[['count','min','25%','50%','75%','max']]
    
    num_summary_sorted = num_summary.sort_values(by=['max','min'], ascending=[False, True]).iloc[:20]

    num_summary_sorted = num_summary_sorted.applymap(lambda x: f"{x:.2e}")

    print("=== Numerical Feature Summary (sorted) ===")
    print(num_summary_sorted)

if __name__ == '__main__':
    main()