from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.models.model import Model, categorical, floats, ints, bools


class LogisticRegressor(Model):
    """
    Logistic Regression with One-Hot Encoding for Categorical Variables
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs.pop('name', None)

        # Define categorical columns
        categorical_columns = categorical
        numerical_columns = floats + ints + bools

        # Handle missing values

        # Define the steps in the pipeline
        # Step 1: OneHotEncoder for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_transformer = SimpleImputer(strategy='median')

        # Use ColumnTransformer to apply the transformations to the correct columns in the dataframe.
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_columns),
                ('num', numerical_transformer, numerical_columns)
            ]
        )

        # Step 2: Logistic Regression
        classifier = LogisticRegression(*args, **kwargs)

        # Create the pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

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
