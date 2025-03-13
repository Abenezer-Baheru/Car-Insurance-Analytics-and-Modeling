# 🚗📊 Car-Insurance-Analytics-and-Modeling

## 🌟 Overview
This project involves analyzing and visualizing insurance data to identify trends, calculate losses and profits, and generate insightful visualizations. The project is divided into several tasks, including data summarization, univariate analysis, bivariate/multivariate analysis, data comparison, and visualizations.

## 📖 Table of Contents
- [🌟 Overview](#-overview)
- [📖 Table of Contents](#-table-of-contents)
- [🗂 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🛠 Data Preprocessing](#-data-preprocessing)
- [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [📊 Hypothesis Testing](#-hypothesis-testing)
- [🔮 Predictive Modeling](#-predictive-modeling)
- [📉 Visualizations](#-visualizations)
- [✅ Unit Tests](#-unit-tests)
- [🤝 Contributing](#-contributing)
- [👤 Author](#-author)
- [📜 License](#-license)

## 🗂 Project Structure

```plaintext
Car-Insurance-Analytics-and-Modeling/
├── 📂 data/                     # Raw and processed data files
├── 📓 notebooks/                # Jupyter notebooks for data exploration and analysis
│   ├── 📜 __init__.py
│   ├── 📝 data_preprocessing.ipynb   # Notebook for data preprocessing
│   ├── 📊 EDA.ipynb                  # Notebook for exploratory data analysis
│   ├── 🔍 AB_hypothesis_testing.py   # Notebook for A/B hypothesis testing
│   ├── 📃 README.md                  # Notebook-specific documentation
│   ├── 📈 statistical_modeling.py    # Notebook for statistical modeling
├── 📂 scripts/                  # Python scripts for data preprocessing, analysis, and modeling
│   ├── 📜 __init__.py
│   ├── 🛠 data_preprocessing.py       # Script for data preprocessing
│   ├── 📊 AB_hypothesis_testing.py    # Script for hypothesis testing
│   ├── 🔮 statistical_modeling.py     # Script for predictive modeling
├── ✅ tests/                    # Unit test scripts for validation
│   ├── ✅ test_AB_hypothesis_testing.py 
│   ├── ✅ test_data_preprocessing.py  # Unit tests for data preprocessing
│   ├── ✅ test_statistical_modeling.py # Unit tests for statistical modeling
├── 🗒️ logs/                     # Logs generated during processing
├── 📄 requirements.txt          # Python package dependencies
├── 📜 README.md                 # Project documentation
├── 📜 LICENSE                   # License file
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Abenezer-Baheru/Car-Insurance-Analytics-and-Modeling.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Car-Insurance-Analytics-and-Modeling
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
1. Run the scripts in sequence to perform data analysis, hypothesis testing, and modeling:
   ```bash
   python preprocess_data.py  # Preprocess the dataset
   python eda.py              # Perform exploratory data analysis
   python hypothesis_test.py  # Conduct hypothesis testing
   python model_training.py   # Train predictive models
   ```

## 🛠 Data Preprocessing
The data preprocessing step includes:
- Handling missing values (imputation, removal).
- Converting columns to appropriate data types.
- Cleaning data for inconsistencies (e.g., trimming white spaces in text).
- Feature engineering to create useful predictors for modeling.

## 📈 Exploratory Data Analysis (EDA)
EDA involves:
- **Univariate Analysis**: Plotting histograms, bar charts, and boxplots for numerical and categorical data.
- **Bivariate/Multivariate Analysis**: Calculating correlations, visualizing scatter plots, and analyzing geographic trends.
- **Outlier Detection**: Using boxplots to identify and handle outliers.
- **Summary Statistics**: Providing descriptive statistics for key columns.

## 📊 Hypothesis Testing
This phase addresses business-specific hypotheses to evaluate:
- Risk differences by geography and demographics.
- Profit margin differences across client categories.
- Statistical relationships between features, using techniques such as t-tests and chi-squared tests.

## 🔮 Predictive Modeling
Predictive models are built to:
- Analyze historical claims data.
- Predict premiums and total claims.
- Models used include:
  - Linear Regression
  - Random Forest
  - Gradient Boosting Machines (XGBoost)
  - Logistic Regression
  - Neural Networks (MLP, RNN/LSTM)
- Model performance is evaluated using metrics like RMSE, accuracy, and feature importance analysis.

## 📉 Visualizations
Visualizations play a crucial role in deriving insights. Examples include:
- Bar plots for profit/loss by cover type or make.
- Heatmaps for correlations between features.
- Line plots to analyze monthly trends in premiums and claims.
- Scatter plots for client segmentation.

## ✅ Unit Tests
Unit testing ensures the reliability of the analysis and modeling steps. Tests include:
- Verifying data types and transformations.
- Testing imputation functions for missing values.
- Validating the accuracy of profit/loss calculations.
- Ensuring proper handling of categorical encoding and scaling.

## 🤝 Contributing
Contributions are welcome! Here's how you can contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Describe your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

## 👤 Author
**Abenezer Baheru**  
GitHub: [Abenezer Baheru](https://github.com/Abenezer-Baheru/Car-Insurance-Analytics-and-Modeling)

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---