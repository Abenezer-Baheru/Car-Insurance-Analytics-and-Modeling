import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

# Sample data for testing
data = pd.DataFrame({
    'TotalPremium': [1000, 2000, 1500, 3000, 2500],
    'TotalClaims': [800, 1800, 1200, 2500, 2000],
    'TransactionMonth': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
    'VehicleIntroDate': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'CoverType': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1'],
    'make': ['Make1', 'Make2', 'Make1', 'Make2', 'Make1'],
    'PostalCode': ['12345', '12345', '67890', '67890', '12345'],
    'MainCrestaZone': ['Zone1', 'Zone2', 'Zone1', 'Zone2', 'Zone1']
})

# Convert columns to appropriate data types
data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])
data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'])

# Calculate the difference between TotalPremium and TotalClaims for each record
data['Difference'] = data['TotalPremium'] - data['TotalClaims']

# Define a function to calculate losses and profits for a given column and show top N results
def calculate_losses(column_name, top_n=None, print_results=False):
    premium_claims = data.groupby(column_name)[['TotalPremium', 'TotalClaims']].sum().reset_index()
    premium_claims['Difference'] = premium_claims['TotalPremium'] - premium_claims['TotalClaims']
    premium_claims['Profit/Loss %'] = (premium_claims['Difference'] / premium_claims['TotalPremium']) * 100
    frequency_counts = data[column_name].value_counts().reset_index()
    frequency_counts.columns = [column_name, 'Frequency']
    frequency_counts['Frequency Percentage %'] = (frequency_counts['Frequency'] / frequency_counts['Frequency'].sum()) * 100
    premium_claims = pd.merge(premium_claims, frequency_counts, on=column_name)
    premium_claims['Avg Profit/Loss per Frequency $'] = premium_claims['Difference'] / premium_claims['Frequency']
    premium_claims = premium_claims.sort_values(by='Avg Profit/Loss per Frequency $', ascending=True)
    if top_n:
        premium_claims = premium_claims.head(top_n)
    total_aggregate_difference = premium_claims['Difference'].sum()
    premium_claims.columns = [column_name, 'TotalPremium $', 'TotalClaims $', 'Difference $', 'Profit/Loss %', 'Frequency', 'Frequency Percentage %', 'Avg Profit/Loss per Frequency $']
    premium_claims['TotalPremium $'] = premium_claims['TotalPremium $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['TotalClaims $'] = premium_claims['TotalClaims $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['Difference $'] = premium_claims['Difference $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['Profit/Loss %'] = premium_claims['Profit/Loss %'].apply(lambda x: f"{x:.2f}%")
    premium_claims['Frequency'] = premium_claims['Frequency'].apply(lambda x: f"{x:,}")
    premium_claims['Frequency Percentage %'] = premium_claims['Frequency Percentage %'].apply(lambda x: f"{x:.2f}%")
    premium_claims['Avg Profit/Loss per Frequency $'] = premium_claims['Avg Profit/Loss per Frequency $'].apply(lambda x: f"{x:,.2f}")
    if print_results:
        print(f"Losses and Profits by {column_name}:")
        print(tabulate(premium_claims, headers='keys', tablefmt='pretty', showindex=False))
        print(f"Total Aggregate Difference for {column_name}: ${total_aggregate_difference:,.2f}\n")
    return premium_claims

class TestTask1(unittest.TestCase):
    def test_data_types(self):
        self.assertEqual(data['TransactionMonth'].dtype, np.dtype('datetime64[ns]'))
        self.assertEqual(data['VehicleIntroDate'].dtype, np.dtype('datetime64[ns]'))

    def test_difference_calculation(self):
        expected_difference = [200, 200, 300, 500, 500]
        self.assertListEqual(data['Difference'].tolist(), expected_difference)

    def test_calculate_losses(self):
        result = calculate_losses('CoverType', top_n=2)
        self.assertEqual(result.shape[0], 2)
        self.assertIn('CoverType', result.columns)
        self.assertIn('TotalPremium $', result.columns)
        self.assertIn('TotalClaims $', result.columns)
        self.assertIn('Difference $', result.columns)
        self.assertIn('Profit/Loss %', result.columns)
        self.assertIn('Frequency', result.columns)
        self.assertIn('Frequency Percentage %', result.columns)
        self.assertIn('Avg Profit/Loss per Frequency $', result.columns)

    def test_missing_values(self):
        missing_values = data.isnull().sum().sum()
        self.assertEqual(missing_values, 0)

    def test_negative_values_removal(self):
        negative_values_count = data[(data['TotalPremium'] < 0) | (data['TotalClaims'] < 0)].shape[0]
        self.assertEqual(negative_values_count, 0)

if __name__ == '__main__':
    unittest.main()
