import pytest
import pandas as pd
import numpy as np
from matplotlib.testing.compare import compare_images
from src.plots import prob_hist

def test_log_odds_hist_with_valid_data():
    # Create a DataFrame with random data
    # Sample from a gaussian with mean -10 and std 10
    false_predictor = pd.Series(np.random.randn(100) * 10 - 10)
    # Sample from a gaussian with mean 10 and std 10
    true_predictor = pd.Series(np.random.randn(100) * 10 + 10)
    predictor = pd.concat([false_predictor, true_predictor]).reset_index(drop=True)
    target = pd.Series([0] * 100 + [1] * 100)

    # Call the function with the test data
    fig = prob_hist(predictor, target)

    # Save the figure to a file
    fig.savefig("test_output.png")

    # Compare the output image with a known good image
    assert compare_images("expected_output.png", "test_output.png", tol=0) is None

def test_log_odds_hist_with_empty_data():
    # Create an empty DataFrame
    predictor = pd.Series()
    target = pd.Series()

    # Call the function with the test data and expect a ValueError
    with pytest.raises(ValueError):
        prob_hist(predictor, target)

def test_log_odds_hist_with_mismatched_data():
    # Create a DataFrame with mismatched predictor and target lengths
    predictor = pd.Series(np.random.rand(100))
    target = pd.Series(np.random.randint(0, 2, 50))

    # Call the function with the test data and expect a ValueError
    with pytest.raises(ValueError):
        prob_hist(predictor, target)