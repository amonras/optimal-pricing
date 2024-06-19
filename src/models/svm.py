from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVC

from src.models.preprocessing import preprocessing
from src.models.model import Model


class SupportVectorMachine(Model):
    """
    GradientBoosting with One-Hot Encoding for Categorical Variables.

    This one implements enhanced feature engineering
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs.pop('name', None)

        prep = preprocessing()

        # Step 2: Logistic Regression
        classifier = SVC(probability=True, *args, **kwargs)

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
        categorical_cols = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
            'onehot'].get_feature_names_out()
        numerical_cols = self.pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out()

        return list(categorical_cols) + list(numerical_cols)

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
