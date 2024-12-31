import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import numpy as np

class StatisticalModeling:
    def __init__(self, data):
        """
        Initialize the StatisticalModeling class with the provided data.
        
        Parameters:
        data (pd.DataFrame): The cleaned data as a DataFrame.
        """
        try:
            self.data = data
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.models = {}
            self.results = {}
        except Exception as e:
            print(f"Error initializing data: {e}")

    def prepare_data(self):
        """
        Prepare the data by handling missing values, feature engineering, encoding categorical data, and train-test split.
        """
        try:
            # Feature Engineering
            self.data['ClaimsPerPremium'] = self.data['TotalClaims'] / self.data['TotalPremium']
            self.data['ClaimsPerPremium'] = self.data['ClaimsPerPremium'].replace([np.inf, -np.inf], 0)
            self.data['ClaimsPerPremium'] = self.data['ClaimsPerPremium'].fillna(0)

            # Encoding Categorical Data
            categorical_features = ['Province', 'PostalCode', 'Gender']
            for feature in categorical_features:
                if self.data[feature].nunique() > 10:
                    # Use Label Encoding for features with many unique values
                    encoder = LabelEncoder()
                    self.data[feature] = encoder.fit_transform(self.data[feature])
                else:
                    # Use OneHot Encoding for features with fewer unique values
                    encoder = OneHotEncoder(sparse_output=False)
                    encoded_features = pd.DataFrame(encoder.fit_transform(self.data[[feature]]))
                    encoded_features.columns = encoder.get_feature_names_out([feature])
                    self.data = pd.concat([self.data.drop(feature, axis=1), encoded_features], axis=1)

            # Ensure all columns are numeric
            self.data = self.data.apply(pd.to_numeric, errors='coerce')

            # Handle infinity or extremely large values
            self.data.replace([np.inf, -np.inf], 0, inplace=True)
            self.data = self.data.applymap(lambda x: 0 if np.abs(x) > 1e10 else x)

            # Handle NaN values by filling with zero
            self.data.fillna(0, inplace=True)

            # Train-Test Split
            X = self.data.drop(['TotalPremium', 'TotalClaims'], axis=1)
            y = self.data['TotalPremium']  # or 'TotalClaims' depending on the target variable
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        except Exception as e:
            print(f"Error preparing data: {e}")

    def build_models(self):
        """
        Build and train the models: Linear Regression, Random Forest, and XGBoost.
        """
        try:
            # Linear Regression
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            self.models['Linear Regression'] = lr

            # Random Forest
            rf = RandomForestRegressor(random_state=42)
            rf.fit(self.X_train, self.y_train)
            self.models['Random Forest'] = rf

            # XGBoost
            xgb = XGBRegressor(random_state=42)
            xgb.fit(self.X_train, self.y_train)
            self.models['XGBoost'] = xgb
        except Exception as e:
            print(f"Error building models: {e}")

    def evaluate_models(self):
        """
        Evaluate the models using appropriate metrics for regression.
        """
        try:
            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                self.results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        except Exception as e:
            print(f"Error evaluating models: {e}")

    def feature_importance(self):
        """
        Analyze feature importance using SHAP values.
        """
        try:
            for name, model in self.models.items():
                explainer = shap.Explainer(model, self.X_train)
                shap_values = explainer(self.X_test)
                shap.summary_plot(shap_values, self.X_test, show=False)
                shap.save_html(f'shap_summary_{name}.html', shap_values)
        except Exception as e:
            print(f"Error analyzing feature importance: {e}")

    def report_results(self):
        """
        Report the comparison between each model's performance.
        """
        try:
            for name, metrics in self.results.items():
                print(f"Model: {name}")
                print(f"MSE: {metrics['MSE']}")
                print(f"MAE: {metrics['MAE']}")
                print(f"R2: {metrics['R2']}")
                print("-" * 30)
        except Exception as e:
            print(f"Error reporting results: {e}")

# Example usage
if __name__ == "__main__":
    try:
        data = pd.read_csv('../src/data/cleaned_data.csv', low_memory=False)
        model = StatisticalModeling(data)
        model.prepare_data()
        model.build_models()
        model.evaluate_models()
        model.feature_importance()
        model.report_results()
        print("Modeling complete.")
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
