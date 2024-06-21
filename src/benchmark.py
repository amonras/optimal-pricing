from pathlib import Path
from typing import List, Dict
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from tqdm.auto import tqdm

from src.models.model import Model

warnings.filterwarnings('ignore', 'ConvergenceWarning: lbfgs failed to converge (status=1)')


class Benchmark:
    def __init__(self, models: Dict[str, Model] = None, data=None):
        if data is None:
            file_path = Path(__file__).parent.parent / 'data/export.parquet'
            self.df = pd.read_parquet(file_path)
        else:
            self.df = data

        # Define features and target variable
        self.X = self.df.drop(['Bag_Purchased'], axis=1)
        self.y = self.df['Bag_Purchased']

        self.models = models

    def evaluate(self, splits=5):
        """
        Evaluate all models on the test set. Compute:
        - ROC-AUC score for each model
        - Expected revenue for each model
        - Confidence intervals for expected revenue

        :return: DataFrame with the evaluation results
        """
        roc_auc = {}
        exp_rev = {}
        conf_int = {}

        # Split the data into training and test sets using 5-fold cross-validation
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

        self.y_pred_prob = pd.DataFrame()
        for name, model in (pbar := tqdm(self.models.items())):
            pbar.set_description(f'Training model {name}')
            # Store the predicted probabilities for each model

            # Initialize the columns for the model
            self.y_pred_prob[name] = pd.Series(index=self.y.index)
            for train_index, test_index in tqdm(skf.split(self.X, self.y), 'Cross-validation', leave=False,
                                                total=splits):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                model.fit(X_train, y_train)
                self.y_pred_prob.loc[y_test.index, name] = pd.Series(model.predict_proba(X_test)[:, 1],
                                                                     index=y_test.index)

            roc_auc[name] = roc_auc_score(self.y, self.y_pred_prob[name])

            # Calculate expected revenue
            df_test = self.X.copy()
            df_test['Bag_Purchased'] = self.y
            df_test['Pred_Prob'] = self.y_pred_prob[name]
            df_test['Expected_Revenue'] = df_test['Pred_Prob'] * df_test['bag_total_price']
            exp_rev[name] = df_test['Expected_Revenue'].sum()

        return pd.DataFrame({"ROC-AUC Score": roc_auc}).sort_values(by="ROC-AUC Score", ascending=False)

    def roc_auc_plot(self, models=None):
        """
        Plot the ROC curve for each model.

        :param y_pred_prob: DataFrame with predicted probabilities for each model
        """
        models = models if models is not None else self.models.keys()

        y_pred_prob = self.y_pred_prob

        plt.figure(figsize=(10, 6))
        for model in y_pred_prob[models].columns:
            fpr, tpr, _ = roc_curve(self.y, y_pred_prob[model])
            roc_auc = roc_auc_score(self.y, y_pred_prob[model])
            plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
