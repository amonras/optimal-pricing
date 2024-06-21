from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_evaluate(data):
    from src.benchmark import Benchmark
    from src.models.baseline import LogisticRegressor
    from src.models.logistic_regression import LogisticRegressor2

    model1 = LogisticRegressor()
    model2 = LogisticRegressor2(preprocessing__imputer=True)
    benchmark = Benchmark(
        models={
            'm1': model1,
            'm2': model2
        },
        data=data,
    )

    benchmark.evaluate()


bool_vars = [False, True]


@pytest.mark.parametrize(
    "imputer,engineered,interactions,svd,scaler",
    np.array(np.meshgrid(*([bool_vars] * 5))).T.reshape(-1, 5)
)
def test_preprocessing(data, imputer, engineered, interactions, svd, scaler):
    from src.models.preprocessing import build_model

    X = data.drop(['Bag_Purchased'], axis=1)
    y = data['Bag_Purchased']

    svd = False
    pipeline = build_model(
        imputer=imputer,
        engineered=engineered,
        interactions=interactions,
        dim_reduction=svd,
        scaler=scaler
    )

    pipeline.fit(X, y)

    assert 'bag_base_price' not in pipeline.steps[0][1].get_feature_names_out()
    assert 'bag_total_price' in pipeline.steps[-1][1].get_feature_names_out()

    if engineered:
        assert 'bag_base_price' not in list(pipeline.named_steps['engineered'].get_feature_names_out())
        assert 'bag_total_price' in list(pipeline.named_steps['engineered'].get_feature_names_out())
        # Ensure price_ratio and log_gdp_ratio are included
        assert 'price_ratio' in list(pipeline.named_steps['engineered'].get_feature_names_out())
        assert 'log_grp_ratio' in list(pipeline.named_steps['engineered'].get_feature_names_out())

    if interactions:
        # Ensured linear terms are preserved
        assert 'price' in list(pipeline.named_steps['interactions'].get_feature_names_out())
        # Ensured quadratic terms are included
        assert 'price^2' in list(pipeline.named_steps['interactions'].get_feature_names_out())
