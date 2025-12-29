import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


class Dataset:
    def __init__(self, filepath, target,
                 drop_columns=None, drop_if_null=None, fill_mean_if_null=None,
                 mode='train', seed=123, normalize=True, y_div_factor=100000.0,
                 integer_categorical_columns=None, onehot_columns=None):
        """
        Args:
            filepath: CSV path
            target: name of target column
            drop_columns: list of columns to drop entirely
            drop_if_null: list of columns for which rows with NaN are dropped
            fill_mean_if_null: list of columns for which NaN are filled with mean
            mode: 'train' or 'test'
            seed: random seed
            normalize: whether to standardize numerical features
            integer_categorical_columns: list of categorical columns to keep as integer labels
            onehot_columns: list of categorical columns to one-hot encode
        """
        

        np.random.seed(seed)
        self.mode = mode
        self.y_div_factor = y_div_factor

        drop_columns = drop_columns or []
        drop_if_null = drop_if_null or []
        fill_mean_if_null = fill_mean_if_null or []
        integer_categorical_columns = integer_categorical_columns or []
        onehot_columns = onehot_columns or []

        # Load CSV
        df = pd.read_csv(filepath)

        # Drop unwanted columns
        df = df.drop(columns=drop_columns, errors='ignore')

        # Drop rows with NaN in specific columns
        if drop_if_null:
            df = df.dropna(subset=drop_if_null)

        # Fill NaN with mean for numeric columns
        for col in fill_mean_if_null:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Separate target
        y = df[target].values / self.y_div_factor
        X = df.drop(columns=[target])

        # Detect numerical columns automatically
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        # Identify categorical columns for each type
        self.integer_categorical_features = [c for c in integer_categorical_columns if c in X.columns]
        self.onehot_features = [c for c in onehot_columns if c in X.columns]

        # Remaining categorical columns (not integer, not one-hot)
        remaining_cats = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.auto_onehot_features = [c for c in remaining_cats
                                     if c not in self.integer_categorical_features
                                     and c not in self.onehot_features]

        # Preprocessing pipelines
        num_steps = [('imputer', SimpleImputer(strategy='mean'))]
        if normalize:
            num_steps.append(('scaler', StandardScaler()))
        num_transformer = Pipeline(num_steps)

        cat_onehot_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        cat_integer_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))
        ])

        # ColumnTransformer
        transformers = []
        if self.numerical_features:
            transformers.append(('num', num_transformer, self.numerical_features))
        if self.onehot_features + self.auto_onehot_features:
            transformers.append(('onehot', cat_onehot_transformer, self.onehot_features + self.auto_onehot_features))
        if self.integer_categorical_features:
            transformers.append(('integer_cat', cat_integer_transformer, self.integer_categorical_features))

        self.preprocessor = ColumnTransformer(transformers)

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True
        )

        # Fit transformer
        self.X_train = self.preprocessor.fit_transform(X_train)
        self.X_test = self.preprocessor.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

        # Feature names
        feature_names = []
        if self.numerical_features:
            feature_names += self.numerical_features
        if self.onehot_features + self.auto_onehot_features:
            onehot_names = self.preprocessor.named_transformers_['onehot'] \
                .named_steps['onehot'].get_feature_names_out(self.onehot_features + self.auto_onehot_features)
            feature_names += onehot_names.tolist()
        if self.integer_categorical_features:
            feature_names += self.integer_categorical_features

        self.feature_names = np.array(feature_names)
        self.n_features = self.X_train.shape[1]

        # Assign x, y according to mode
        if mode == 'train':
            self.x = self.X_train
            self.y = self.y_train
        else:
            self.x = self.X_test
            self.y = self.y_test
