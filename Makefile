.PHONY: setup eda model-selection optimization test

# Setup the Python environment
setup:
	pip install -e .

# Run the main script
eda:
	# open jupyter notebook EDA.ipynb
	jupyter notebook EDA.ipynb

model-selection:
	# open jupyter notebook Model_Selection.ipynb
	jupyter notebook Model\ Selection.ipynb

optimization:
	# open jupyter notebook Optimal Pricing.ipynb
	jupyter notebook Optimal\ Pricing.ipynb

# Run the tests
test:
	pytest test/