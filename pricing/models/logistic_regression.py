from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

from pricing.models.preprocessing import build_model
from pricing.models.features import ExtraFeatures
from pricing.models.model import Model, categorical, floats, ints, bools


class LogisticRegressor2(Model):
    """
    Logistic Regression with One-Hot Encoding for Categorical Variables.

    This one implements enhanced feature engineering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs.pop('name', None)

        # Extract all preprocessing arguments
        preprocessing_kwargs = {}
        keys, vals = zip(*kwargs.items())
        for k, v in zip(keys, vals):
            if 'preprocessing__' in k:
                preprocessing_kwargs[k.split('preprocessing__')[1]] = v
                kwargs.pop(k)

        # Step 1: Preprocessing
        prep = build_model(**preprocessing_kwargs)

        # Step 2: Logistic Regression
        classifier = LogisticRegression(*args, **kwargs)

        # Create the pipeline
        self.pipeline = Pipeline(steps=prep.steps + [('classifier', classifier)])

    def optimal_price(self, X):
        """
        Calculate the optimal price for each observation in X.

        :param X: DataFrame with features
        :return: Series with the optimal price for each observation
        """
        if self.model is None:
            raise ValueError('Model has not been trained yet.')

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError('Model has not been trained yet.')
        return self.model.coef_

    def get_feature_names(self):
        if self.pipeline is None:
            raise ValueError('Pipeline has not been trained yet.')

        return self.pipeline.steps[-2][1].get_feature_names_out()

    def model_coefficients(self):
        """
        Return the coefficients of the logistic regression model
        :return:
        """
        if self.model is None:
            raise ValueError('Model has not been trained yet.')

        return {k: v for k, v in zip(self.get_feature_names(), self.model.coef_[0][:])}

    def gamma(self):
        """
        Return the coefficient of 'bag_total_price' feature
        :return:
        """
