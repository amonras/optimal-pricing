import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.baseline import LogisticRegressor
# Load the dataset
file_path = '../data/export.parquet'
df = pd.read_parquet(file_path)

# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Preprocessing
# Encode categorical variables if needed
# For simplicity, let's assume no encoding is needed at this step

# Define features and target variable
X = df.drop(['Bag_Purchased'], axis=1)
y = df['Bag_Purchased']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegressor()
model.fit(X_train, y_train)

# Predict probabilities on the test set
y_pred_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f'ROC-AUC Score: {roc_auc}')

# Calculate expected revenue
df_test = X_test.copy()
df_test['Bag_Purchased'] = y_test
df_test['Pred_Prob'] = y_pred_prob
df_test['Expected_Revenue'] = df_test['Pred_Prob'] * df_test['bag_total_price']
expected_revenue = df_test['Expected_Revenue'].sum()
print(f'Expected Revenue: {expected_revenue}')

# Bootstrap to estimate confidence intervals
n_iterations = 1000
bootstrap_revenues = []

for _ in range(n_iterations):
    bootstrap_sample = df_test.sample(n=len(df_test), replace=True)
    bootstrap_revenue = bootstrap_sample['Expected_Revenue'].sum()
    bootstrap_revenues.append(bootstrap_revenue)

bootstrap_revenues = np.array(bootstrap_revenues)
confidence_interval = np.percentile(bootstrap_revenues, [2.5, 97.5])
print(f'95% Confidence Interval for Expected Revenue: {confidence_interval}')

# Plot the bootstrap distribution
sns.histplot(bootstrap_revenues, kde=True)
plt.axvline(confidence_interval[0], color='red', linestyle='--')
plt.axvline(confidence_interval[1], color='red', linestyle='--')
plt.title('Bootstrap Distribution of Expected Revenue')
plt.xlabel('Expected Revenue')
plt.ylabel('Frequency')
plt.show()
