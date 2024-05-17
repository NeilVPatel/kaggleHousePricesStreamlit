import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from io import StringIO

# Cache the data to avoid reloading it multiple times
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# Cache the correlation matrix to avoid recomputing it
@st.cache_data
def compute_correlations(df):
    return df.corr()

def display_eda(train):
    st.title('Exploratory Data Analysis')

    # Dataset Overview
    st.subheader('Dataset Overview')
    st.write(train.head(5))  # Display only the first 10 rows
    with st.expander("Show Code"):
        st.code("""
# Load the dataset
train.head(5)
        """, language='python')
    st.write("We can see that a lot of the data is categorical and describes the properties of the house/lot.")

    # Summary Statistics
    st.subheader('Summary Statistics')
    st.write(train.describe(include='all').head(10))  # Display summary statistics for first 10 columns
    with st.expander("Show Code"):
        st.code("""
# Summary statistics
train.describe(include='all').head(10)
        """, language='python')

    # Data Types and Missing Values
    st.subheader('Data Types and Missing Values')
    buffer = StringIO()
    train.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    with st.expander("Show Code"):
        st.code("""
# Data types and missing values
train.info(buf=buffer)
info_str = buffer.getvalue()
st.text(info_str)
        """, language='python')
    st.write("Only 3 different datatypes are present: integer, floating point numbers, and objects. There are a few columns that are missing a lot of values.")

    # Percentage of Missing Values
    st.subheader('Percentage of Missing Values')
    missing_percentage = train.isnull().sum() * 100 / len(train)
    missing_percentage_df = pd.DataFrame(missing_percentage, columns=['Missing_Percentage'])
    missing_percentage_df = missing_percentage_df.sort_values("Missing_Percentage", ascending=False).query('Missing_Percentage != 0')
    st.write(missing_percentage_df)
    with st.expander("Show Code"):
        st.code("""
# Percentage of missing values
missing_percentage = train.isnull().sum() * 100 / len(train)
missing_percentage_df = pd.DataFrame(missing_percentage, columns=['Missing_Percentage'])
missing_percentage_df = missing_percentage_df.sort_values("Missing_Percentage", ascending=False).query('Missing_Percentage != 0')
st.write(missing_percentage_df)
        """, language='python')
    st.write("Only 19 columns have missing values, and only 4 of those have greater than 80% missing values.")

    # Histograms for Numerical Columns
    st.subheader('Histograms for Numerical Columns')
    num_columns = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
    num_cols = 3  # Number of columns for the subplot grid
    num_rows = (len(num_columns) + num_cols - 1) // num_cols  # Calculate number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, column in zip(axes, num_columns):
        sns.histplot(train[column], kde=True, ax=ax)
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')

    # Remove any empty subplots
    for i in range(len(num_columns), len(axes)):
        fig.delaxes(axes[i])

    st.pyplot(fig)
    with st.expander("Show Code"):
        st.code("""
# Histograms for numerical columns as a grid
num_columns = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]
num_cols = 3  # Number of columns for the subplot grid
num_rows = (len(num_columns) + num_cols - 1) // num_cols  # Calculate number of rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 5))
axes = axes.flatten()  # Flatten the axes array for easy iteration

for ax, column in zip(axes, num_columns):
    sns.histplot(train[column], kde=True, ax=ax)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

# Remove any empty subplots
for i in range(len(num_columns), len(axes)):
    fig.delaxes(axes[i])

st.pyplot(fig)
        """, language='python')

    st.subheader('Correlation Matrix')
    int_cols = train[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                    'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'YrSold', 'SalePrice']]
    correlations = compute_correlations(int_cols)

    # Display the correlation matrix
    st.dataframe(correlations.style.background_gradient(cmap='coolwarm'))

    with st.expander("Show Code"):
        st.code("""
# Correlation matrix
int_cols = train[['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MoSold', 'YrSold', 'SalePrice']]
correlations = int_cols.corr()
st.dataframe(correlations.style.background_gradient(cmap='coolwarm'))
        """, language='python')