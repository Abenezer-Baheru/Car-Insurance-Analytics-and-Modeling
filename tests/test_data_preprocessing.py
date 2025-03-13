import unittest
import pandas as pd
import sys
import os

# Adjust the path to include the directory where AB_hypothesis_testing.py is located
sys.path.append(os.path.abspath('../script'))

from data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a sample dataset for testing."""
        cls.test_file = "test_data.csv"
        cls.sample_data = pd.DataFrame({
            'TotalPremium': [100, 200, -50, 300],
            'TotalClaims': [50, -100, 200, 400],
            'NewVehicle': [None, 'Yes', 'No', None],
            'Bank': [None, 'ABC Bank', None, 'XYZ Bank'],
            'AccountType': [None, 'Savings', None, 'Current'],
            'Gender': ['Male', None, 'Female', 'Male'],
            'MaritalStatus': [None, 'Single', 'Married', 'Single'],
            'mmcode': ['A1', None, 'B2', 'C3'],
            'VehicleType': [None, 'SUV', 'Sedan', 'Truck'],
            'make': [None, 'Toyota', 'Honda', 'Ford'],
            'VehicleIntroDate': [None, '2010-01-01', None, '2020-01-01'],
            'NumberOfDoors': [4, None, 4, 2],
            'bodytype': [None, 'SUV', 'Sedan', 'Truck'],
            'kilowatts': [None, 100, None, 150],
            'cubiccapacity': [2000, None, 1800, 2500],
            'Cylinders': [None, 4, 6, 8],
            'Model': [None, 'Corolla', 'Civic', 'F-150'],
            'CapitalOutstanding': [None, 5000, 10000, 15000]
        })
        cls.sample_data.to_csv(cls.test_file, index=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test dataset."""
        if os.path.exists(cls.test_file):
            os.remove(cls.test_file)

    def setUp(self):
        """Initialize the DataPreprocessor object."""
        self.preprocessor = DataPreprocessor(file_path=self.test_file)

    def test_load_data(self):
        """Test loading of the data."""
        self.preprocessor.load_data()
        self.assertIsNotNone(self.preprocessor.data)
        self.assertEqual(len(self.preprocessor.data), 4)

    def test_handle_missing_values(self):
        """Test handling of missing values."""
        self.preprocessor.load_data()
        self.preprocessor.handle_missing_values()
        self.assertFalse(self.preprocessor.data.isnull().any().any())

    def test_remove_negative_values(self):
        """Test removal of negative values."""
        self.preprocessor.load_data()
        self.preprocessor.remove_negative_values()
        self.assertTrue((self.preprocessor.data['TotalPremium'] >= 0).all())
        self.assertTrue((self.preprocessor.data['TotalClaims'] >= 0).all())

    def test_strip_whitespace(self):
        """Test stripping of whitespace."""
        self.preprocessor.load_data()
        self.preprocessor.strip_whitespace()
        self.assertTrue(self.preprocessor.data.applymap(
            lambda x: x.strip() if isinstance(x, str) else x).equals(self.preprocessor.data))

    def test_save_cleaned_data(self):
        """Test saving of the cleaned data."""
        cleaned_file = "cleaned_data.csv"
        self.preprocessor.load_data()
        self.preprocessor.save_cleaned_data(cleaned_file)
        self.assertTrue(os.path.exists(cleaned_file))
        os.remove(cleaned_file)

if __name__ == "__main__":
    unittest.main()