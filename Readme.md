# Optimal Pricing Strategy

## How to run the code
1. Clone the repository
2. If you want to use a virtual environment, create one and activate it
3. Run `make setup`

To make sure that everything is working correctly, you can run the tests by running `make test`.

If you want to run the code in the jupyter notebooks, make sure to install your virtual environment as a kernel
by running `ipython kernel install --user --name=<your-virtual-env>`

This code has been developed and tested with Python 3.12

## Exploratory Data Analysis
The [`EDA.ipynb`](EDA.ipynb) notebook performs an Exploratory Data Analysis (EDA) on a dataset. It analyzes
the different variable distributions, their missing values and the correlation between them.

You can open the notebook by running `make eda`.

## Model Selection
The [`Model_Selection.ipynb`](Model_Selection.ipynb) notebook explores different data preprocessing techniques such as different
imputations methods, encoding, feature engineering and feature selection. It also explores different regression
models and finds that the best model is a Gradient Boosting Regressor with engineered features.

You can open the notebook by running `make model-selection`.

## Price Optimization
The [`Optimal_Pricing.ipynb`](Optimal_Pricing.ipynb) notebook uses a simple Logistic Regression model to estimate the 
customer demand and its response to price changes. The notebook uses this model to compute the optimal price
and explore different pricing strategies. For two different strategies, the expected revenue and demand, 
and their volatilities are computed.

<img src="images/results_constrained.png" alt="Summary of results" width="300"/>

You can open the notebook by running `make optimization`.

### Optimal Pricing
The optimal pricing strategy is obtained by using the standard optimal price equation which relates marginal cost
to marginal revenue. The Logistic Regression makes it simple to compute demand elasticity.

### Constrained Optimal Pricing
An alternative pricing strategy is explored where the optimization is performed given a constraint on the 
resulting demand, which should be no less than a given threshold.

### Volatility analysis
The Bootstrap method is used to verify the expected confidence intervals for the revenue and demand estimates.

<img src="images/revenue_histograms.png" alt="Bootstrap" width="300"/>