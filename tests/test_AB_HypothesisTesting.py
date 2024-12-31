import sys
import os
import unittest
import pandas as pd

# Adjust the path to include the directory where AB_hypothesis_testing.py is located
sys.path.append(os.path.abspath('../script'))

from AB_hypothesis_testing import ABHypothesisTesting

class TestABHypothesisTesting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up a sample DataFrame for testing and initialize the ABHypothesisTesting class.
        """
        data = {
            'Province': ['A', 'B', 'A', 'B'],
            'PostalCode': ['X', 'Y', 'X', 'Y'],
            'Gender': ['Male', 'Female', 'Female', 'Male'],
            'TotalClaims': [10, 20, 30, 40],
            'TotalPremium': [100, 200, 300, 400]
        }
        cls.df = pd.DataFrame(data)
        cls.analyzer = ABHypothesisTesting(cls.df)

    def test_chi_squared_test_province(self):
        """
        Test the chi_squared_test method for the Province feature.
        """
        try:
            p_value = self.analyzer.chi_squared_test('Province', 'TotalClaims')
            self.assertIsNotNone(p_value)
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)
        except Exception as e:
            self.fail(f"test_chi_squared_test_province raised an exception: {e}")

    def test_chi_squared_test_postal_code(self):
        """
        Test the chi_squared_test method for the PostalCode feature.
        """
        try:
            p_value = self.analyzer.chi_squared_test('PostalCode', 'TotalClaims')
            self.assertIsNotNone(p_value)
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)
        except Exception as e:
            self.fail(f"test_chi_squared_test_postal_code raised an exception: {e}")

    def test_chi_squared_test_gender(self):
        """
        Test the chi_squared_test method for the Gender feature.
        """
        try:
            p_value = self.analyzer.chi_squared_test('Gender', 'TotalClaims')
            self.assertIsNotNone(p_value)
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)
        except Exception as e:
            self.fail(f"test_chi_squared_test_gender raised an exception: {e}")

    def test_t_test(self):
        """
        Test the t_test method for the PostalCode feature.
        """
        try:
            group_a_data = self.df[self.df['PostalCode'] == 'X']['TotalPremium']
            group_b_data = self.df[self.df['PostalCode'] == 'Y']['TotalPremium']
            if len(group_a_data) < 2 or len(group_b_data) < 2:
                self.skipTest("Sample size too small for t-test")
            p_value = self.analyzer.t_test('PostalCode', 'PostalCode', 'TotalPremium')
            self.assertIsNotNone(p_value)
            self.assertGreaterEqual(p_value, 0)
            self.assertLessEqual(p_value, 1)
        except Exception as e:
            self.fail(f"test_t_test raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
