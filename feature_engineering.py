import streamlit as st

def display_feature_engineering():
    st.title('Feature Engineering')

    # Explanation of Feature Engineering Steps
    st.subheader('Feature Engineering Steps')
    st.markdown("""
    In this section, I outline the steps taken to engineer new features from the existing data. The following steps were performed:
    
    1. **Creating a Bed and Bath Ratio:**
        - A new feature `Bed_bath_Ratio` was created by dividing the number of bedrooms above grade by the number of full baths above grade.
    
    2. **Calculating Lot Area to Ground Living Area Ratio:**
        - A new feature `lot_vs_grlivarea` was created by dividing the `LotArea` by the `GrLivArea`.
    
    3. **Calculating Combined Square Footage:**
        - A new feature `combinedsqft` was created by summing the `1stFlrSF`, `2ndFlrSF`, and `TotalBsmtSF`.

    """)

    # Code for Feature Engineering in an expandable field
    with st.expander("Show Feature Engineering Code"):
        st.code("""
# Feature Engineering Steps
def feature_engineering(train, test):
    # Create Bed and Bath Ratio
    train['Bed_bath_Ratio'] = train['BedroomAbvGr'] / (train['FullBath'] + train['HalfBath']*0.5)
    test['Bed_bath_Ratio'] = test['BedroomAbvGr'] / (test['FullBath'] + test['HalfBath']*0.5)

    # Calculate Lot Area to Ground Living Area Ratio
    train['lot_vs_grlivarea'] = train['LotArea'] / train['GrLivArea']
    test['lot_vs_grlivarea'] = test['LotArea'] / test['GrLivArea']

    # Calculate Combined Square Footage
    train['combinedsqft'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']
    test['combinedsqft'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']

    return train, test

# Apply feature engineering
train, test = feature_engineering(train, test)
        """, language='python')