import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtraFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)

        return self

    def transform(self, X):
        out = pd.DataFrame(
            dict(
                price_ratio=X['bag_total_price'] / X['price'],
                log_grp_ratio=X['src_dst_gdp']
            ),
            index=X.index)
        result = pd.concat([X, out], axis=1)

        return result

    def set_output(self, transform="pandas"):
        self.transform = transform
        return self

    def get_new_columns(self):
        return ['price_ratio', 'log_grp_ratio']

    def get_feature_names_out(self, input_features=None):
        input_features = self.feature_names_in_ if input_features is None else input_features
        return np.concatenate([input_features, self.get_new_columns()])
