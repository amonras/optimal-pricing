import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def odds_fn(x):
    if x == 0:
        return 0
    elif x == 1:
        return np.inf
    else:
        return x / (1 - x)


def prob_hist(predictor: pd.Series, target: pd.Series, nbins=20):
    """
    Create a histogram of the different classes as grouped by the target variable.

    On a secondary axis, plot the log-odds of the target variable for each predictor bin
    Superimpose the standard logistic function as a function of the predictor variable

    :param predictor:
    :param target:
    :return:
    """
    # Create a DataFrame with the predictor and target variable
    df = pd.concat([predictor, target], axis=1)
    df.columns = ['predictor', 'target']

    # Create a histogram of the predictor variable, split by the target variable
    fig, ax1 = plt.subplots()

    # Split the predictor variable by the target variable
    groups = [predictor.loc[g].values for g in predictor.groupby(target).groups.values()]

    ax1.hist(groups, label=target, bins=nbins)

    ax1.set_xlabel('odds')
    ax1.set_ylabel('Counts')

    # Create a secondary axis with the frequency of the target variable
    ax2 = ax1.twinx()
    grouped = df.groupby([pd.cut(df['predictor'], nbins), target], observed=False)['target']

    count = grouped.count().unstack()
    freq = count.apply(lambda x: x / x.sum(), axis=1)
    error = np.sqrt(freq[True] * freq[False] / count.sum(axis=1)).replace({0: np.nan})

    # Calculate the mid-point of each interval
    mid_points = [interval.mid for interval in freq.index]

    # Use mid_points as x-values in the plot
    ax2.errorbar(mid_points, freq[True], yerr=error, color='r', marker='o', linestyle='none', capsize=5)

    # Superimpose the standard logistic function
    x = np.linspace(df['predictor'].min(), df['predictor'].max(), 100)
    y = 1 / (1 + np.exp(-x))
    ax2.plot(x, y, color='g', linestyle='--')

    return fig

