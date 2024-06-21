from sklearn.impute._base import _BaseImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class LinearRegressionImputer(_BaseImputer):
    """
    Impute missing values using a Linear Regression model for each category in a categorical variable.
    If the category is not present in the training data, a global Linear Regression model is used.
    If the numerical predictor has nans, the target inherits the nan value.

    Does not accept nans in the predictor variables.

    Parameters
    ----------
    target : str
        Name of the target variable
    numerical : str
        Name of the numerical variable
    categorical : str
        Name of the categorical variable
    """
    def __init__(self, target, numerical, categorical):
        self.target = target
        self.numerical = numerical
        self.categorical = categorical
        self.models = {}
        self.overall_model = LinearRegression()
        self.le = None

    def fit(self, df, *args, **kwargs):
        for category, mask in df.groupby(self.categorical).groups.items():
            model = LinearRegression()
            x = df.loc[mask][[self.numerical]]
            y = df.loc[mask][self.target]
            model.fit(x[y.notna()], y[y.notna()])
            self.models[category] = model

        x = df[[self.numerical]]
        y = df[self.target]
        self.overall_model.fit(x[y.notna()], y[y.notna()])

        return self

    def transform(self, df):
        for category, mask in df.groupby(self.categorical).groups.items():
            model = self.models.get(category, self.overall_model)

            x = df.loc[mask][[self.numerical]].dropna()
            predictions = pd.Series(model.predict(x), index=x.index)
            df.loc[predictions.index, self.target] = predictions
        return df

    def set_output(self, transform="pandas"):
        self.transform = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.get_feature_names_in

    def get_params(self, deep=True):
        return {
            'target': self.target,
            'numerical': self.numerical,
            'categorical': self.categorical
        }