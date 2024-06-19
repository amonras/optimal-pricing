import pandas as pd
from _pytest.fixtures import fixture


@fixture
def data():
    return pd.read_parquet('test.parquet')
