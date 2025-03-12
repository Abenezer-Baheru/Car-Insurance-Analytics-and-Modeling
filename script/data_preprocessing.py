import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,  # Set the minimum logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/data_preprocessing.log"),  # Save logs to the specified folder
        logging.StreamHandler()                                          # Also output logs to the console
    ]
)

class DataPreprocessor:
    def __init__(self, file_path, delimiter='|'):
        """Initialize the preprocessor with data file details."""
        self.file_path = file_path
        self.delimiter = delimiter
        self.data = None

    def load_data(self):
        """Load the dataset."""
        try:
            self.data = pd.read_csv(self.file_path, delimiter=self.delimiter)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def summarize_data(self):
        """Provide a summary of the dataset."""
        if self.data is not None:
            print(self.data.info())
            print(self.data.head())
            print(self.data.describe())
            logging.info("Data summary provided.")
        else:
            logging.warning("Data has not been loaded yet.")

    def check_duplicates(self):
        """Check for duplicated rows and columns."""
        if self.data is not None:
            duplicate_rows = self.data.duplicated().sum()
            duplicate_columns = self.data.columns[self.data.columns.duplicated()].tolist()
            print(f"Duplicated rows: {duplicate_rows}")
            print(f"Duplicated columns: {duplicate_columns}")
            logging.info(f"Checked duplicates. Rows: {duplicate_rows}, Columns: {duplicate_columns}")
        else:
            logging.warning("Data has not been loaded yet.")

    def handle_missing_values(self):
        """Handle missing values by removing or imputing."""
        if self.data is not None:
            # Remove columns with high missing percentages
            columns_to_remove = ['NumberOfVehiclesInFleet', 'CrossBorder', 'CustomValueEstimate',
                                 'WrittenOff', 'Converted', 'Rebuilt']
            self.data.drop(columns=columns_to_remove, inplace=True, errors='ignore')
            
            # Fill missing values
            self.data['NewVehicle'].fillna(self.data['NewVehicle'].mode()[0], inplace=True)
            self.data['Bank'].fillna('Unknown', inplace=True)
            self.data['AccountType'].fillna('Unknown', inplace=True)
            
            # Remove rows with missing values in critical columns
            critical_columns = ['Gender', 'MaritalStatus', 'mmcode', 'VehicleType', 'make',
                                'VehicleIntroDate', 'NumberOfDoors', 'bodytype', 'kilowatts',
                                'cubiccapacity', 'Cylinders', 'Model', 'CapitalOutstanding']
            self.data.dropna(subset=critical_columns, inplace=True)
            logging.info("Missing values handled.")
        else:
            logging.warning("Data has not been loaded yet.")

    def remove_negative_values(self):
        """Remove rows with negative values in 'TotalPremium' or 'TotalClaims'."""
        if self.data is not None:
            negative_values_count = self.data[(self.data['TotalPremium'] < 0) | (self.data['TotalClaims'] < 0)].shape[0]
            print(f"Negative rows before removal: {negative_values_count}")
            self.data = self.data[(self.data['TotalPremium'] >= 0) & (self.data['TotalClaims'] >= 0)]
            logging.info(f"Removed rows with negative values. Total removed: {negative_values_count}")
        else:
            logging.warning("Data has not been loaded yet.")

    def strip_whitespace(self):
        """Remove leading and trailing whitespaces from string columns."""
        if self.data is not None:
            self.data = self.data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            logging.info("Leading and trailing whitespaces removed.")
        else:
            logging.warning("Data has not been loaded yet.")

    def outlier_handling(self):
        """Handle outliers for 'TotalPremium' and 'TotalClaims'."""
        if self.data is not None:
            # Create boxplots for visualization
            def plot_selected_boxplots(data, columns):
                data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), figsize=(15, 6), sharex=False, sharey=False)
                plt.tight_layout()
                plt.show()

            print("Boxplots for 'TotalPremium' and 'TotalClaims':")
            plot_selected_boxplots(self.data, ['TotalPremium', 'TotalClaims'])

            # Define quantile thresholds
            lower_quantile = 0.01
            upper_quantile = 0.99

            # Calculate quantile values for 'TotalPremium' and 'TotalClaims'
            total_premium_lower = self.data['TotalPremium'].quantile(lower_quantile)
            total_premium_upper = self.data['TotalPremium'].quantile(upper_quantile)
            total_claims_lower = self.data['TotalClaims'].quantile(lower_quantile)
            total_claims_upper = self.data['TotalClaims'].quantile(upper_quantile)

            # Count rows that would be removed
            rows_to_remove = self.data[
                (self.data['TotalPremium'] < total_premium_lower) | (self.data['TotalPremium'] > total_premium_upper) |
                (self.data['TotalClaims'] < total_claims_lower) | (self.data['TotalClaims'] > total_claims_upper)
            ].shape[0]

            print(f"TotalPremium lower quantile (1%): {total_premium_lower}")
            print(f"TotalPremium upper quantile (99%): {total_premium_upper}")
            print(f"TotalClaims lower quantile (1%): {total_claims_lower}")
            print(f"TotalClaims upper quantile (99%): {total_claims_upper}")
            print(f"Number of rows that would be removed: {rows_to_remove}")

            # Summary of 'TotalClaims'
            print("\nSummary of 'TotalClaims':")
            total_claims_summary = self.data['TotalClaims'].describe()
            print(total_claims_summary)

            # Calculate specific percentiles
            total_claims_95th = self.data['TotalClaims'].quantile(0.95)
            total_claims_99th = self.data['TotalClaims'].quantile(0.9972)
            print(f"95th percentile of 'TotalClaims': {total_claims_95th}")
            print(f"99.72th percentile of 'TotalClaims': {total_claims_99th}")

            # Count the number of positive values in 'TotalClaims'
            total_claims_greater_than_zero = self.data[self.data['TotalClaims'] > 0].shape[0]
            print(f"Number of values greater than zero in 'TotalClaims': {total_claims_greater_than_zero}")

            # Function to calculate IQR and count outliers
            def count_outliers_iqr(data, column):
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
                return outliers.shape[0]

            # Count outliers for 'TotalPremium' and 'TotalClaims'
            total_premium_outliers = count_outliers_iqr(self.data, 'TotalPremium')
            total_claims_outliers = count_outliers_iqr(self.data, 'TotalClaims')

            print(f"Number of outliers in 'TotalPremium': {total_premium_outliers}")
            print(f"Number of outliers in 'TotalClaims': {total_claims_outliers}")
            logging.info("Outlier handling completed.")
        else:
            logging.warning("Data has not been loaded yet.")

    def save_cleaned_data(self, save_path):
        """Save the cleaned dataset."""
        if self.data is not None:
            self.data.to_csv(save_path, index=False)
            logging.info(f"Cleaned data saved to {save_path}")
            print(f"Cleaned data saved to {save_path}")
        else:
            logging.warning("Data has not been loaded yet.")