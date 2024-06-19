from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from models.imputer import LinearRegressionImputer
from src.models.features import ExtraFeatures
from src.models.model import categorical, floats, ints, bools


def preprocessing(*args, **kwargs):
    """
    Define the preprocessing pipeline for the model
    :param derived_features:
    :param interaction_terms:
    :param svd:
    :return:
    """

    # Define categorical columns
    categorical_columns = categorical
    numerical_columns = floats + ints + bools

    # Define the steps in the pipeline
    # Step 1: Impute missing values
    lr_imputer_cols = ['price', 'travel_time', 'booking_window_group']
    imputer = ColumnTransformer(
        transformers=[
            ('price', LinearRegressionImputer('price', 'travel_time', 'booking_window_group'), lr_imputer_cols),
            ('num', SimpleImputer(strategy='median'), [c for c in numerical_columns if c not in lr_imputer_cols]),
            ('cat', SimpleImputer(strategy='most_frequent'),
             [c for c in categorical_columns if c not in lr_imputer_cols])
        ],
        remainder='drop',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # Step 2: OneHotEncoder for categorical columns
    ohe = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # Step 3: Generate some derived features
    engineered = ExtraFeatures().set_output(transform="pandas")
    new_columns = engineered.get_new_columns()

    engineered_features = ColumnTransformer(
        transformers=[
            ('precompute', engineered, ['bag_total_price', 'price', 'src_dst_gdp'])
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # Step 4: Generate interaction terms
    interactions = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)

    # Step 5: Dimensionality reduction
    svd = TruncatedSVD(n_components=30)

    steps = [
        ('imputer', imputer),
        ('ohe', ohe),
        ('engineered', engineered_features),
        ('interactions', interactions),
        ('svd', svd),
        ('standardscaler', StandardScaler())
    ]

    pipeline = Pipeline(steps=[step for step in steps if kwargs.get(step[0], True)])

    return pipeline
