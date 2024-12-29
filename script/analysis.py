# Import libraries for data manipulation and analysis
import pandas as pd
import numpy as np

# Import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nbformat
from tabulate import tabulate

# Import libraries for statistical analysis
from scipy import stats

# Import libraries for machine learning
from sklearn.linear_model import LinearRegression

# Set visualization styles
sns.set(style="whitegrid")
plt.style.use('ggplot')  # or use 'seaborn-dark'

# Data Summarization
# Load the data with the correct delimiter
data = pd.read_csv('../src/data/MachineLearningRating_v3.txt', delimiter='|')

# Display the column names
print(data.columns)

# Head of the data
data.head()

# Calculate descriptive statistics
descriptive_stats = data[['TotalPremium', 'TotalClaims']].describe()
print(descriptive_stats)

# Check data types
print(data.dtypes)

# Display the first few rows of the TransactionMonth and VehicleIntroDate columns
print(data[['TransactionMonth', 'VehicleIntroDate']].head())

# Convert columns to appropriate data types
data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'])
data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'])

# Print the data types of TransactionMonth and VehicleIntroDate
print(data['TransactionMonth'].dtype)
print(data['VehicleIntroDate'].dtype)

# Check the size of the data without including the index column
data_size = data.shape[1] - 1  # Subtract 1 for the index column
print(f'The dataset contains {data.shape[0]} rows and {data_size} columns (excluding the index column).')

# Check for duplicated rows
duplicated_rows = data.duplicated()
print(f'Total duplicated rows: {duplicated_rows.sum()}')

# Check for duplicated columns
duplicated_columns = data.columns[data.columns.duplicated()]
print(f'Duplicated columns: {duplicated_columns}')

# Calculate the percentage of missing values for each column
missing_percentage = data.isnull().sum() / len(data) * 100

# Filter columns with missing values and sort them in descending order
missing_columns_sorted = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
print(missing_columns_sorted)

# Remove columns with high missing percentages
columns_to_remove = ['NumberOfVehiclesInFleet', 'CrossBorder', 'CustomValueEstimate', 'WrittenOff', 'Converted', 'Rebuilt']
data = data.drop(columns=columns_to_remove)

# Fill missing values for specific columns
data['NewVehicle'].fillna(data['NewVehicle'].mode()[0], inplace=True)
data['Bank'].fillna('Unknown', inplace=True)
data['AccountType'].fillna('Unknown', inplace=True)
data['Citizenship'].fillna('Unknown', inplace=True)

# Remove rows with missing values in specific columns
columns_to_check = ['Gender', 'MaritalStatus', 'mmcode', 'VehicleType', 'make', 'VehicleIntroDate', 'NumberOfDoors', 'bodytype', 'kilowatts', 'cubiccapacity', 'Cylinders', 'Model', 'CapitalOutstanding']
data = data.dropna(subset=columns_to_check)

# Verify the changes
missing_values_after = data.isnull().sum()
print(missing_values_after)

# Check for leading or trailing whitespaces in the values of each column
whitespace_values = {}
for column in data.columns:
    if data[column].dtype == 'object':  # Only check for string columns
        whitespace_values[column] = data[column].str.contains(r'^\s|\s$').sum()

# Display columns with leading or trailing whitespaces in their values
for column, count in whitespace_values.items():
    if count > 0:
        print(f'Column "{column}" has {count} values with leading or trailing whitespaces.')

# Remove leading and trailing whitespaces from all columns
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Check the size of the data without including the index column
data_size = data.shape[1] - 1  # Subtract 1 for the index column
print(f'The dataset contains {data.shape[0]} rows and {data_size} columns (excluding the index column).')

# Function to create a single plot with box plots for specified columns
def plot_selected_boxplots(data, columns):
    data[columns].plot(kind='box', subplots=True, layout=(1, len(columns)), figsize=(15, 6), sharex=False, sharey=False)
    plt.tight_layout()
    plt.show()

# Create a single plot with box plots for 'TotalPremium' and 'TotalClaims'
plot_selected_boxplots(data, ['TotalPremium', 'TotalClaims'])

# Count rows with negative values in 'TotalPremium' or 'TotalClaims'
negative_values_count = data[(data['TotalPremium'] < 0) | (data['TotalClaims'] < 0)].shape[0]
print(f"Number of rows with negative values in 'TotalPremium' or 'TotalClaims': {negative_values_count}")

# Remove rows with negative values in 'TotalPremium' or 'TotalClaims'
data = data[(data['TotalPremium'] >= 0) & (data['TotalClaims'] >= 0)]

# Count rows with negative values in 'TotalPremium' or 'TotalClaims'
negative_values_count = data[(data['TotalPremium'] < 0) | (data['TotalClaims'] < 0)].shape[0]
print(f"Number of rows with negative values in 'TotalPremium' or 'TotalClaims': {negative_values_count}")

# Univariate Analysis
# Plot histograms for numerical columns
numerical_columns = ['TotalPremium', 'TotalClaims']
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# Plot bar charts for categorical columns in descending order
categorical_columns = ['CoverType', 'make']
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data[column], order=data[column].value_counts().index)
    plt.title(f'Distribution of {column}')
    if column == 'make':
        plt.yticks(fontsize=6, fontweight='bold')  # Decrease font size and make y-axis labels bold for 'make'
    plt.show()

# Bivariate or Multivariate Analysis
# Plot scatter plot for TotalPremium vs TotalClaims by PostalCode
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', data=data)
plt.title('TotalPremium vs TotalClaims by PostalCode')
plt.show()

# Calculate correlation matrix for TotalPremium and TotalClaims
correlation_matrix = data[['TotalPremium', 'TotalClaims']].corr()

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Calculate monthly changes for TotalPremium and TotalClaims
data['TotalPremiumChange'] = data.groupby('PostalCode')['TotalPremium'].diff()
data['TotalClaimsChange'] = data.groupby('PostalCode')['TotalClaims'].diff()

# Plot scatter plot for TotalPremiumChange vs TotalClaimsChange by PostalCode
plt.figure(figsize=(10, 6))
sns.scatterplot(x='TotalPremiumChange', y='TotalClaimsChange', hue='PostalCode', data=data)
plt.title('Monthly Change in TotalPremium vs TotalClaims by PostalCode')
plt.show()

# Calculate correlation matrix for TotalPremiumChange and TotalClaimsChange
correlation_matrix = data[['TotalPremiumChange', 'TotalClaimsChange']].corr()

# Plot correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Monthly Changes')
plt.show()

# Data Comparison
# Calculate monthly changes for TotalPremium and TotalClaims
data['TotalPremiumChange'] = data.groupby('MainCrestaZone')['TotalPremium'].diff()
data['TotalClaimsChange'] = data.groupby('MainCrestaZone')['TotalClaims'].diff()

# Calculate the count of each CoverType and make per MainCrestaZone
cover_type_counts = data.groupby('MainCrestaZone')['CoverType'].value_counts().unstack().fillna(0)
make_counts = data.groupby('MainCrestaZone')['make'].value_counts().unstack().fillna(0)

# Sort the legend items alphabetically
cover_type_counts = cover_type_counts[sorted(cover_type_counts.columns)]
make_counts = make_counts[sorted(make_counts.columns)]

# Create a custom color palette with 45 distinct colors
custom_palette = sns.color_palette("hsv", 45)

# Plot the trends for CoverType using a horizontal bar chart
plt.figure(figsize=(12, 8))
cover_type_counts.plot(kind='barh', stacked=True, color=custom_palette)
plt.title('Trends in Insurance Cover Type by MainCrestaZone')
plt.ylabel('MainCrestaZone', fontweight='normal')
plt.xlabel('Count')
plt.legend(title='CoverType', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot the trends for make with the legend at the bottom, in 6 columns, decreased font size, unbolded y-axis label, and increased figure size
plt.figure(figsize=(20, 14))  # Increase the figure height
make_counts.plot(kind='barh', stacked=True, color=custom_palette)
plt.title('Trends in Auto Make by MainCrestaZone')
plt.ylabel('MainCrestaZone', fontweight='normal', fontsize=8)
plt.xlabel('Count', fontsize=12)
plt.legend(title='Make', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')  # Adjust legend position
plt.tight_layout()
plt.show()

# Plot the trends for TotalPremiumChange by MainCrestaZone
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='TransactionMonth', y='TotalPremiumChange', hue='MainCrestaZone', palette=custom_palette)
plt.title('Monthly Change in TotalPremium by MainCrestaZone')
plt.xlabel('TransactionMonth')
plt.ylabel('TotalPremiumChange')
plt.legend(title='MainCrestaZone', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()

# Plot the trends for TotalClaimsChange by MainCrestaZone
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='TransactionMonth', y='TotalClaimsChange', hue='MainCrestaZone', palette=custom_palette)
plt.title('Monthly Change in TotalClaims by MainCrestaZone')
plt.xlabel('TransactionMonth')
plt.ylabel('TotalClaimsChange')
plt.legend(title='MainCrestaZone', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.show()


# Calculate the difference between TotalPremium and TotalClaims for each record
data['Difference'] = data['TotalPremium'] - data['TotalClaims']

# Define a function to calculate losses and profits for a given column and show top N results
def calculate_losses(column_name, top_n=None, print_results=False):
    # Group by the specified column and calculate the sum of TotalPremium, TotalClaims, and Difference
    premium_claims = data.groupby(column_name)[['TotalPremium', 'TotalClaims']].sum().reset_index()
    premium_claims['Difference'] = premium_claims['TotalPremium'] - premium_claims['TotalClaims']
    
    # Calculate the profit/loss percentage for each group
    premium_claims['Profit/Loss %'] = (premium_claims['Difference'] / premium_claims['TotalPremium']) * 100
    
    # Calculate the frequency and frequency percentage
    frequency_counts = data[column_name].value_counts().reset_index()
    frequency_counts.columns = [column_name, 'Frequency']
    frequency_counts['Frequency Percentage %'] = (frequency_counts['Frequency'] / frequency_counts['Frequency'].sum()) * 100
    
    # Merge the frequency counts with the premium claims
    premium_claims = pd.merge(premium_claims, frequency_counts, on=column_name)
    
    # Calculate the average profit/loss per frequency
    premium_claims['Avg Profit/Loss per Frequency $'] = premium_claims['Difference'] / premium_claims['Frequency']
    
    # Sort by Avg Profit/Loss per Frequency $
    premium_claims = premium_claims.sort_values(by='Avg Profit/Loss per Frequency $', ascending=True)
    
    # Select top N results if specified
    if top_n:
        premium_claims = premium_claims.head(top_n)
    
    # Calculate the total aggregate difference
    total_aggregate_difference = premium_claims['Difference'].sum()
    
    # Format the columns
    premium_claims.columns = [column_name, 'TotalPremium $', 'TotalClaims $', 'Difference $', 'Profit/Loss %', 'Frequency', 'Frequency Percentage %', 'Avg Profit/Loss per Frequency $']
    premium_claims['TotalPremium $'] = premium_claims['TotalPremium $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['TotalClaims $'] = premium_claims['TotalClaims $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['Difference $'] = premium_claims['Difference $'].apply(lambda x: f"{x:,.2f}")
    premium_claims['Profit/Loss %'] = premium_claims['Profit/Loss %'].apply(lambda x: f"{x:.2f}%")
    premium_claims['Frequency'] = premium_claims['Frequency'].apply(lambda x: f"{x:,}")
    premium_claims['Frequency Percentage %'] = premium_claims['Frequency Percentage %'].apply(lambda x: f"{x:.2f}%")
    premium_claims['Avg Profit/Loss per Frequency $'] = premium_claims['Avg Profit/Loss per Frequency $'].apply(lambda x: f"{x:,.2f}")
    
    # Print the results if specified
    if print_results:
        print(f"Losses and Profits by {column_name}:")
        print(tabulate(premium_claims, headers='keys', tablefmt='pretty', showindex=False))
        print(f"Total Aggregate Difference for {column_name}: ${total_aggregate_difference:,.2f}\n")
    
    return premium_claims

# List of columns to analyze
columns_to_analyze = [
    'LegalType', 'TransactionMonth', 'CoverType', 'CoverCategory', 'CoverGroup', 'Section', 'Product', 'Province', 
    'MainCrestaZone', 'SubCrestaZone', 'IsVATRegistered', 'Citizenship', 'Title', 'MaritalStatus', 'Gender', 
    'Bank', 'AccountType', 'VehicleType', 'RegistrationYear', 'make', 'bodytype', 'AlarmImmobiliser', 
    'NewVehicle', 'TermFrequency', 'ExcessSelected'
]

# Example of running the analysis for a specific column without printing
# You can call this function for each column as needed
# result = calculate_losses('CoverType', top_n=10, print_results=False)

# Generate table results and visualizations for each column
for column in columns_to_analyze:
    result = calculate_losses(column, top_n=10, print_results=True)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    sns.barplot(x=column, y='Difference $', data=result, palette='muted')
    plt.title(f'Total Profit/Loss by {column}')
    plt.xlabel(column)
    plt.ylabel('Total Profit/Loss ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=column, y='Frequency Percentage %', data=result, palette='muted')
    plt.title(f'Frequency Percentage by {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency Percentage (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=column, y='Avg Profit/Loss per Frequency $', data=result, palette='muted')
    plt.title(f'Average Profit/Loss per Frequency by {column}')
    plt.xlabel(column)
    plt.ylabel('Avg Profit/Loss per Frequency ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

