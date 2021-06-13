from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p


class all_purpose_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        X_new = X.copy()
        self.cols = ['MSZoning', 'SaleType', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Electrical']
        self.mode_to_replace = {col: X_new[col].mode()[0] for col in self.cols}

        return self

    def transform(self, X):
        X_new = X.copy()

        # impute with NA/None
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            X_new[col] = X_new[col].fillna('NA')
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
            X_new[col] = X_new[col].fillna('NA')
        X_new['FireplaceQu'].fillna('NA', inplace=True)
        X_new['MasVnrType'].fillna('None', inplace=True)
        X_new['Alley'].fillna('NA', inplace=True)
        X_new['PoolQC'].fillna('NA', inplace=True)
        X_new['Fence'].fillna('NA', inplace=True)
        X_new['MiscFeature'].fillna('NA', inplace=True)

        # impute with 0
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            X_new[col] = X_new[col].fillna(0)
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
            X_new[col] = X_new[col].fillna(0)

        X_new['MasVnrArea'].fillna(0, inplace=True)

        # impute with mode
        for col in self.cols:
            X_new.loc[X_new[col].isna(), col] = self.mode_to_replace[col]

            # other
        X_new['Functional'].fillna('Typ', inplace=True)
        X_new['TotalSF'] = X_new['TotalBsmtSF'] + X_new['1stFlrSF'] + X_new['2ndFlrSF']
        X_new['MSSubClass'] = X_new['MSSubClass'].astype('object')

        X_new.drop(['Utilities', 'Id'], axis=1, inplace=True)

        return X_new


class skewness_remover(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        X_new = X.copy()
        self.skewness = X_new[X_new.describe().columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        self.skewness = self.skewness[self.skewness > 0.75]

        return self

    def transform(self, X):
        X_new = X.copy()
        for col in self.skewness.index:
            X_new[col] = boxcox1p(X_new[col], 0.15)

        return X_new


class custom_imputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        X_new = X.copy()
        self.median_to_replace = X_new.groupby('Neighborhood')['LotFrontage'].median()

        return self

    def transform(self, X):
        X_new = X.copy()
        for n, value in zip(self.median_to_replace.index, self.median_to_replace.values):
            X_new.loc[(X_new['LotFrontage'].isna()) & (X_new['Neighborhood'] == n), 'LotFrontage'] = value

        return X_new