import pandas as pd
import numpy as np
from matplotlib.testing.compare import compare_images
from pricing.plots import prob_hist


def test_log_odds_hist_with_valid_data():
    """
    We use this test to fine tune the histogram plot with simple data
    """
    # Create a DataFrame with random data
    np.random.seed(0)

    # Sample from a gaussian with mean -10 and std 10
    false_predictor = pd.Series(np.random.randn(100) * 10 - 10)
    # Sample from a gaussian with mean 10 and std 10
    true_predictor = pd.Series(np.random.randn(100) * 10 + 10)
    predictor = pd.concat([false_predictor, true_predictor]).reset_index(drop=True)
    target = pd.Series([False] * 100 + [True] * 100)

    # Call the function with the test data
    fig = prob_hist(predictor, target)

    # Save the figure to a file
    fig.savefig("test/test_output.png")

    # Compare the output image with a known good image
    assert compare_images("test/expected_output.png", "test/test_output.png", tol=0) is None
