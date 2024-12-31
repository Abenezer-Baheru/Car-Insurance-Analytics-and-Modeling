import unittest
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../script'))
from statistical_modeling import StatisticalModeling

class TestStatisticalModeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the cleaned data
        cleaned_data_path = '../src/data/cleaned_data.csv'
        cls.data = pd.read_csv(cleaned_data_path, low_memory=False)
        cls.model = StatisticalModeling(cls.data)
    
    def test_prepare_data(self):
        # Test the prepare_data method
        self.model.prepare_data()
        self.assertIsNotNone(self.model.X_train)
        self.assertIsNotNone(self.model.X_test)
        self.assertIsNotNone(self.model.y_train)
        self.assertIsNotNone(self.model.y_test)
    
    def test_build_models(self):
        # Test the build_models method
        self.model.prepare_data()
        self.model.build_models()
        self.assertIn('Linear Regression', self.model.models)
        self.assertIn('Random Forest', self.model.models)
        self.assertIn('XGBoost', self.model.models)
    
    def test_evaluate_models(self):
        # Test the evaluate_models method
        self.model.prepare_data()
        self.model.build_models()
        self.model.evaluate_models()
        self.assertIn('Linear Regression', self.model.results)
        self.assertIn('Random Forest', self.model.results)
        self.assertIn('XGBoost', self.model.results)
    
    def test_feature_importance(self):
        # Test the feature_importance method
        self.model.prepare_data()
        self.model.build_models()
        self.model.feature_importance()
        # Check if SHAP summary plots are generated (this is a placeholder check)
        self.assertTrue(True)
    
    def test_report_results(self):
        # Test the report_results method
        self.model.prepare_data()
        self.model.build_models()
        self.model.evaluate_models()
        self.model.report_results()
        # Check if results are printed (this is a placeholder check)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
