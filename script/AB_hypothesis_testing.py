# AB_hypothesis_testing.py

import pandas as pd
import scipy.stats as stats

class ABHypothesisTesting:
    def __init__(self, data):
        """
        Initialize the ABHypothesisTesting class with the provided data.
        
        Parameters:
        data (pd.DataFrame): The dataset to be used for hypothesis testing.
        """
        self.data = data

    def chi_squared_test(self, feature, target):
        """
        Perform a chi-squared test to evaluate the relationship between a categorical feature and a target variable.
        
        Parameters:
        feature (str): The name of the categorical feature.
        target (str): The name of the target variable.
        
        Returns:
        float: The p-value of the chi-squared test.
        """
        try:
            contingency_table = pd.crosstab(self.data[feature], self.data[target])
            chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
            return p_value
        except Exception as e:
            print(f"Error performing chi-squared test for {feature} and {target}: {e}")
            return None

    def t_test(self, group_a, group_b, target):
        """
        Perform a t-test to compare the means of a target variable between two groups.
        
        Parameters:
        group_a (str): The name of the first group.
        group_b (str): The name of the second group.
        target (str): The name of the target variable.
        
        Returns:
        float: The p-value of the t-test.
        """
        try:
            group_a_data = self.data[self.data[group_a] == 'A'][target]
            group_b_data = self.data[self.data[group_b] == 'B'][target]
            if len(group_a_data) < 2 or len(group_b_data) < 2:
                raise ValueError("Sample size too small for t-test")
            t_stat, p_value = stats.ttest_ind(group_a_data, group_b_data)
            return p_value
        except Exception as e:
            print(f"Error performing t-test for {group_a} and {group_b} on {target}: {e}")
            return None