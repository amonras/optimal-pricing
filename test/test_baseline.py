from pricing.models.baseline import LogisticRegressor


def test_model_coefficients(data):
    X = data.drop(['Bag_Purchased'], axis=1)
    y = data['Bag_Purchased']

    model = LogisticRegressor()

    model.fit(X, y)

    assert 'bag_base_price' not in model.model_coefficients()
    assert 'bag_total_price' in model.model_coefficients()
