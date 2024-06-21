from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler, FunctionTransformer

from pricing.models.imputer import LinearRegressionImputer
from pricing.models.features import ExtraFeatures, log_cols
from pricing.models.model import categorical, floats, ints, bools, airlines, booking_window_group, partner, market_group


def build_model(
        *args,
        imputer=False,
        ohe=True,
        engineered=False,
        interactions=False,
        dim_reduction=False,
        scaler=False,
        features=None,
        model=None,
        **kwargs
):
    """
    Define the preprocessing pipeline for the model
    :param derived_features:
    :param interaction_terms:
    :param dim_reduction:
    :return:
    """

    # Define categorical columns
    categorical_columns = categorical
    numerical_columns = floats + ints + bools

    steps = []

    # Define the steps in the pipeline
    # Step 1: Impute missing values

    # Specifically tailored imputer for the 'price' column because it is the one with most missing values
    if imputer:
        lr_imputer_cols = ['price', 'travel_time', 'booking_window_group']
    else:
        lr_imputer_cols = []

    # Impute the rest of the numerical and categorical columns
    num_imputer_cols = [c for c in numerical_columns
                        if c not in lr_imputer_cols
                        # Careful about 'bag_base_price' because
                        # it would prevent from extrapolating
                        # to different prices
                        and c != 'bag_base_price']
    cat_imputer_cols = [c for c in categorical_columns if c not in lr_imputer_cols]
    transformers = [
        ('cat', SimpleImputer(strategy='most_frequent'), cat_imputer_cols),
        ('num', SimpleImputer(strategy='median'), num_imputer_cols)
    ]
    if imputer:
        transformers.append(
            ('price', LinearRegressionImputer('price', 'travel_time', 'booking_window_group'), lr_imputer_cols),
        )

    imputer_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop other features ('bag_base_price')
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    steps.append(('imputer', imputer_transformer))

    # Step 2: OneHotEncoder for categorical columns
    if ohe:
        ohe_transformer = ColumnTransformer(
            transformers=[
                (
                    'onehot',
                    OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False,
                        categories=[airlines, booking_window_group, partner, market_group]
                    ),
                    categorical_columns)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
        steps.append(('ohe', ohe_transformer))

    # Step 3: Generate some derived features
    if engineered:
        engineered_transformer = ColumnTransformer(
            transformers=[
                ('precompute', ExtraFeatures(), ['bag_total_price'] + log_cols)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
        numerical_columns.extend(ExtraFeatures.get_new_columns())
        steps.append(('engineered', engineered_transformer))

    # Step 4: Generate interaction terms
    if interactions:
        interactions_transformer = ColumnTransformer(
            transformers=[
                ('num', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False), numerical_columns)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
        steps.append(('interactions', interactions_transformer))

    # Step 7: Scale the features
    if scaler:
        scaler_transformer = StandardScaler().set_output(transform="pandas")
        steps.append(('scaler', scaler_transformer))

    # Step 5: Dimensionality reduction
    if dim_reduction == "svd":
        svd_transformer = TruncatedSVD(n_components=30)
        steps.append(('svd', svd_transformer))
    if dim_reduction == "kbest":
        kbest_transformer = SelectKBest(k=30)
        steps.append(('kbest', kbest_transformer))
    if dim_reduction == 'pca':
        pca_transformer = PCA(n_components=30)
        steps.append(('pca', pca_transformer))

    # Step 6: Select specific features:
    if features is not None:
        feature_selector = ColumnTransformer(
            transformers=[
                # Generate the identity transformation for each feature
                ('select', FunctionTransformer(lambda X: X), features)
            ],
            remainder='drop',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
        steps.append(('select', feature_selector))

    if model is not None:
        steps.append(('classifier', model))

    pipeline = Pipeline(steps=steps)
    return pipeline
