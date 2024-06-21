
categorical = ['airlines', 'booking_window_group', 'partner', 'market_group']
floats = ['distance', 'est_dst_temperature', 'src_dst_gdp', 'bag_total_price', 'travel_time', 'price']
ints = ['children', 'bag_weight', 'nr_of_stopovers', 'passengers']
bools = ['is_intercontinental', 'within_country', 'us_movement_outside_us']

airlines = [f'airline{i}' for i in range(1, 14)]
booking_window_group = ['14 - 20 days', '4 - 6 days', '0 - 3 days', '21 - 60 days',
       '7 - 13 days', '61 - 121 days', '122 - 178 days', '179 - 319 days']
partner = ['partner1', 'partner2']
market_group = [f'market{i}' for i in range(1, 19)]

class Model:
    """
    Base class for all models in the project.
    """
    def __init__(self, *args, **kwargs):
        self.name = kwargs.pop('name', None) or 'generic'
        self.description = self.__doc__
        self.pipeline = None
        self.model = None

    def optimal_price(self, X):
        """
        Calculate the optimal price for each observation in X.

        :param X: DataFrame with features
        :return: Series with the optimal price for each observation
        """
        raise NotImplementedError

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline.named_steps['classifier']
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def get_feature_importance(self):
        if self.model is None:
            raise ValueError('Model has not been trained yet.')
        return self.model.coef_

    def get_feature_names(self):
        if self.pipeline is None:
            raise ValueError('Pipeline has not been trained yet.')
        return self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()

    def get_pipeline(self):
        return self.pipeline