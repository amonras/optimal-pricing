import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


log_cols = ['distance', 'price', 'travel_time', 'src_dst_gdp', 'passengers']


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
                # Careful with this feature if we end up computing demand elasticities with it
                price_ratio=X['bag_total_price'] / X['price'],
                log_grp_ratio=X['src_dst_gdp']
            ),
            index=X.index)

        logs = np.log(X[log_cols])
        logs.columns = [f'log_{c}' for c in log_cols]

        result = pd.concat([X, out, logs], axis=1)

        return result

    def set_output(self, transform="pandas"):
        self.transform = transform
        return self

    @staticmethod
    def get_new_columns():
        return ['price_ratio', 'log_grp_ratio'] + [f'log_{c}' for c in log_cols]

    def get_feature_names_out(self, input_features=None):
        input_features = self.feature_names_in_ if input_features is None else input_features
        return np.concatenate([input_features, self.get_new_columns()])
