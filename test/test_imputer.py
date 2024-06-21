import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from pricing.models.imputer import LinearRegressionImputer


def test_imputer_with_known_categories():
    """
    Ensure that the imputer works when all categories are present in the training data
    """
    df_train = pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'numerical': [1, 2, 6, 8, 10],
        'categorical': ['a', 'a', 'b', 'b', 'b']
    })
    df_test = pd.DataFrame({
        'numerical': [2, 6, 8],
        'categorical': ['a', 'b', 'b']
    })
    imputer = LinearRegressionImputer('target', 'numerical', 'categorical')
    imputer.fit(df_train)
    df_test = imputer.transform(df_test)
    assert df_test['target'].tolist() == [2, 3, 4]


def test_imputer_with_unseen_categories():
    """
    Ensure that the imputer works with unseen categories, falling back to the global regression
    """
    df_train = pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'numerical': [1, 2, 3, 4, 5],
        'categorical': ['a', 'a', 'b', 'b', 'b']
    })
    df_test = pd.DataFrame({
        'numerical': [2, 3, 4],
        'categorical': ['a', 'c', 'c']
    })
    imputer = LinearRegressionImputer('target', 'numerical', 'categorical')
    imputer.fit(df_train)
    df_test = imputer.transform(df_test)
    assert df_test['target'].tolist() == [2, 3, 4]


def test_imputer_with_missing_values():
    """
    Ensure that the imputer works with missing values in the predictor, returning missing values in the target
    """
    df_train = pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'numerical': [1, 2, 3, 4, 5],
        'categorical': ['a', 'a', 'b', 'b', 'b']
    })
    df_test = pd.DataFrame({
        'numerical': [2, 3, None],
        'categorical': ['a', 'b', 'b']
    })
    imputer = LinearRegressionImputer('target', 'numerical', 'categorical')
    imputer.fit(df_train)
    df_test = imputer.transform(df_test)
    assert_array_equal(df_test['target'].to_list(),  [2., 3., np.nan])


def test_output_dataframe():
    """
    Ensure that the output of the imputer is a DataFrame with the same shape and columns as the input
    """
    df_train = pd.DataFrame({
        'target': [1, 2, 3, 4, 5],
        'numerical': [1, 2, 3, 4, 5],
        'categorical': ['a', 'a', 'b', 'b', 'b']
    })
    df_test = pd.DataFrame({
        'numerical': [2, 3, 4],
        'categorical': ['a', 'b', 'b']
    })
    imputer = LinearRegressionImputer('target', 'numerical', 'categorical')
    imputer.fit(df_train)
    df_test = imputer.transform(df_test)
    assert isinstance(df_test, pd.DataFrame)
    assert df_test.shape[0] == 3
    assert df_test.shape[1] == 3
    assert 'target' in df_test.columns
    assert 'numerical' in df_test.columns
    assert 'categorical' in df_test.columns
    assert df_test['target'].tolist() == [2, 3, 4]
    assert df_test['numerical'].tolist() == [2, 3, 4]
    assert df_test['categorical'].tolist() == ['a', 'b', 'b']
    assert df_test['target'].dtype == 'float64'
    assert df_test['numerical'].dtype == df_train['numerical'].dtype
    assert df_test['categorical'].dtype == 'object'
    assert df_test.isnull().sum().sum() == 0