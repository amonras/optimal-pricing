# Optimal Pricing Strategy

## How to run the code
1. Clone the repository
2. If you want to use a virtual environment, create one and activate it
3. Run `make setup`

To make sure that everything is working correctly, you can run the tests by running `make test`.

If you want to run the code in the jupyter notebooks, make sure to install your virtual environment as a kernel
by running `ipython kernel install --user --name=<venv>`


## Exploratory Data Analysis
The `EDA.ipynb` notebook performs an Exploratory Data Analysis (EDA) on a dataset. It analyzes
the different variable distributions, their missing values and the correlation between them.

You can view the notebook [here](EDA.ipynb) or by running `make eda`.

## Model Selection
The `Model Selection.ipynb` notebook explores different data preprocessing techniques such as different
imputations methods, encoding, feature engineering and feature selection. It also explores different regression
models and finds that the best model is a Gradient Boosting Regressor with engineered features.

## Price Optimization
The `Optimal Pricing.ipynb` notebook uses a simple Logistic Regression model to estimate the customer demand for
and explore different pricing strategies. The expected revenue and resulting demand from the following strategies 
are computed.

### Optimal Pricing
The optimal pricing strategy is obtained by using the standard optimal price equation which relates marginal cost
to marginal revenue. The Logistic Regression makes it simple to compute demand elasticity.

### Constrained Optimal Pricing
An alternative pricing strategy is explored where the optimization is performed given a constraint on the 
resulting demand, which should be no less than a given threshold.

### Volatility analysis
The Bootstrap method is used to verify the expected confidence intervals for the revenue and demand estimates.