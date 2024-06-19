from pathlib import Path

import pandas as pd


def test_evaluate(data):
    from src.benchmark import Benchmark
    from src.models.baseline import LogisticRegressor
    from src.models.logistic_regression import LogisticRegressor2

    model1 = LogisticRegressor()
    model2 = LogisticRegressor2()
    benchmark = Benchmark(
        models=[
            model1,
            model2
        ],
        data=data
    )

    benchmark.evaluate()


def test_preprocessing(data):
    from src.models.preprocessing import preprocessing

    X = data.drop(['Bag_Purchased'], axis=1)
    y = data['Bag_Purchased']

    pipeline = preprocessing(
        imputer=True,
        ohe=True,
        engineered=True,
        interactions=False,
        svd=False,
        scaler=False
    )

    pipeline.fit(X, y)

    assert 'bag_base_price' not in pipeline.steps[-1][1].get_feature_names_out()
    assert 'bag_total_price' in pipeline.steps[-1][1].get_feature_names_out()

    assert 'bag_base_price' not in list(pipeline.named_steps['engineered'].get_feature_names_out())
    assert 'bag_total_price' in list(pipeline.named_steps['engineered'].get_feature_names_out())
