import streamlit as st

def display_data_cleaning():
    st.title('Data Cleaning')

    # Explanation of Data Cleaning Steps
    st.subheader('Data Cleaning Steps')
    st.markdown("""
    In this section, I outline the steps taken to clean the training dataset. The following steps were performed:
    
    1. **Fill Missing Values for Training Data:**
        - Columns filled with 'NA': `BsmtQual`, `BsmtCond`, `BsmtFinType1`, `BsmtFinType2`, `BsmtExposure`, `GarageQual`, `GarageCond`, `GarageFinish`, `GarageYrBlt`, `GarageType`, `FireplaceQu`
        - Columns filled with 0: `Bed_bath_Ratio`, `MasVnrArea`
        - Columns filled with 'None': `MasVnrType`
        - Columns filled with median value: `LotFrontage`
        - Columns filled with mode value: `Electrical`
        - Special handling for `Bed_bath_Ratio`: Replace `NaN` and `inf` with 0

    """)

    # Code for Data Cleaning in an expandable field
    with st.expander("Show Data Cleaning Code"):
        st.code("""
# Function to clean data
def clean_data(train):
    # Fill in missing values for training data
    to_na = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure',
             'GarageQual', 'GarageCond', 'GarageFinish', 'GarageYrBlt',
             'GarageType', 'FireplaceQu']
    to_zero = ['Bed_bath_Ratio', 'MasVnrArea']
    to_none = ['MasVnrType']
    to_median = ['LotFrontage']
    to_mode = ['Electrical']

    for col in to_na:
        train[col].fillna('NA', inplace=True)
    for col in to_zero:
        train[col].fillna(0, inplace=True)
    for col in to_none:
        train[col].fillna('None', inplace=True)
    median_lotFrontage = train.LotFrontage.median()
    for col in to_median:
        train[col].fillna(median_lotFrontage, inplace=True)
    mode_Electrical = 'SBrkr'
    for col in to_mode:
        train[col].fillna(mode_Electrical, inplace=True)
    train.Bed_bath_Ratio = train.Bed_bath_Ratio.replace([pd.NA, pd.NaT, float('inf')], 0)

    return train
        """, language='python')