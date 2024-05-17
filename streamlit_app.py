# Import libraries
import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostRegressor, Pool
import joblib
import home_page  # Import the home page module
import eda  # Import the EDA module
import data_cleaning    # Import the Data Cleaning module
import feature_engineering  # Import the Feature Engineering module
import model_training  # Import the Model Training module
import conclusion_future_work  # Import the Conclusion and Future Work module
import feature_definitions  # Import the Feature Definitions module

# Path of the trained model, data
MODEL_PATH = "/mount/src/kagglehousepricesstreamlit/catboost_model.cbm"
DATA_PATH = "/mount/src/kagglehousepricesstreamlit/data.parquet"

# Set Page Title
st.set_page_config(page_title="House Prices Project")

# Define data, model, and train/test sets
@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_PATH)
    train = data
    return data, train

@st.cache_data
def load_x_y(file_path):
    data = joblib.load(file_path)
    data.reset_index(drop=True, inplace=True)
    return data

@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    try:
        model.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None
    return model

def calculate_shap(model, X_train, X_test):
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_cat_train = explainer.shap_values(X_train)
    shap_values_cat_test = explainer.shap_values(X_test)
    return explainer, shap_values_cat_train, shap_values_cat_test

def plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, house_id, X_test, X_train):
    house_data = X_test[X_test['house_id'] == house_id]
    print(f"Filtered Data for house_id {house_id}: {house_data}")  # Debugging print

    if not house_data.empty:
        house_index = house_data.index[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.decision_plot(
            base_value=explainer.expected_value, 
            shap_values=shap_values_cat_test[house_index], 
            features=X_test.loc[house_index]
        )

        # Set axis labels
        ax.set_xlabel('Model output value', fontsize=14)
        ax.set_ylabel('Features', fontsize=14)

        # Adjust font size and weight for ticks
        for item in ([ax.xaxis.label, ax.yaxis.label]):
            item.set_fontsize(12)
            item.set_weight('bold')

        # Optionally, you can adjust colors
        plt.axhline(0, color='black', linewidth=0.5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        st.pyplot(fig)
        plt.close()
    else:
        st.error("No house found with the specified ID.")
        return

def display_shap_summary(shap_values_cat_train, X_train):
    # Create the plot summarizing the SHAP values
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", show=False, plot_size=(12, 8))

    # Set axis labels
    ax.set_xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)

    # Adjust font size and weight for ticks
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(12)
        item.set_weight('bold')

    # Optionally, you can adjust colors and add a grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    st.pyplot(fig)
    plt.close()
    
def display_shap_waterfall_plot(explainer, expected_value, shap_values, feature_names, max_display=11):
    # Create SHAP waterfall drawing
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Generate the SHAP waterfall plot
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=feature_names, max_display=max_display, show=False)

    # Set axis labels
    ax.set_xlabel('Model output value', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)

    # Adjust font size and weight for ticks
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(12)
        item.set_weight('bold')

    # Optionally, you can adjust colors and add a grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    st.pyplot(fig)
    plt.close()

def summary(model, data, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Summarize and visualize SHAP values
    st.subheader("Summary SHAP Values Visualization")
    display_shap_summary(shap_values_cat_train, X_train)
    st.caption("This plot shows the distribution of SHAP values for each feature across all houses in the training dataset. Features are ordered by their importance, with the most impactful features at the top.")
    st.caption("Vertical Axis: Lists the features, ordered by their importance. Horizontal Axis: Represents the SHAP values, showing how much each feature contributed to the prediction.")

def plot_shap(model, data, house_id, X_train, X_test):
    # Calculate SHAP values
    explainer, shap_values_cat_train, shap_values_cat_test = calculate_shap(model, X_train, X_test)

    # Visualize SHAP values
    st.subheader("SHAP Values Visualization")
    plot_shap_values(model, explainer, shap_values_cat_train, shap_values_cat_test, house_id, X_test, X_train)
    st.caption("This plot show how each feature influences the model’s prediction for each house. Positive values indicate features that increase the predicted price, while negative values indicate features that decrease it. This helps in understanding which features are most important in the prediction process.")
    
    # Waterfall
    st.subheader("SHAP Waterfall Visualization")
    house_index = X_test[X_test['house_id'] == house_id].index[0]
    display_shap_waterfall_plot(explainer, explainer.expected_value, shap_values_cat_test[house_index], feature_names=X_test.columns, max_display=11)
    st.caption("This plot shows the base value (the average model prediction) and then adds or subtracts the contribution of each feature to show how they lead to the final prediction for a specific house. It gives a step-by-step explanation of how the model arrived at its prediction for that house.")
    
st.title("House Prices Prediction Project")

def main():
    model = load_model()
    data, train = load_data()

    X_train = load_x_y("/mount/src/kagglehousepricesstreamlit/X_train.pkl")
    X_test = load_x_y("/mount/src/kagglehousepricesstreamlit/X_test.pkl")
    y_train = load_x_y("/mount/src/kagglehousepricesstreamlit/y_train.pkl")
    y_test = load_x_y("/mount/src/kagglehousepricesstreamlit/y_test.pkl")

    # Navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to:', [
        'Home',
        'Exploratory Data Analysis',
        'Data Cleaning',
        'Feature Engineering',
        'Model Training and Evaluation',
        'Feature Importance',
        'House-based SHAP Analysis',
        'Conclusion and Future Work',
        'Feature Definitions'
    ])
    
    available_house_ids = X_test['house_id'].tolist()

    # Homepage contents
    if page == 'Home':
        home_page.display_home()
        
    # Call the EDA display function if the selected page is 'Exploratory Data Analysis'
    elif page == 'Exploratory Data Analysis':
        eda.display_eda(train)

    # Call the data_cleaning display function if the selected page is 'Data Cleaning'
    elif page == 'Data Cleaning':
        data_cleaning.display_data_cleaning()

    # Call the feature engineering display function if the selected page is 'Feature Engineering'
    elif page == 'Feature Engineering':
        feature_engineering.display_feature_engineering()

    # Call the model training display function if the selected page is 'Model Training and Evaluation'
    elif page == 'Model Training and Evaluation':
        model_training.display_model_training()

    # If Feature Importance is selected
    elif page == "Feature Importance":
        summary(model, data, X_train=X_train.drop('house_id',axis=1), X_test=X_test.drop('house_id',axis=1))

    # If User-based SHAP option is selected
    elif page == "House-based SHAP Analysis":
        # House ID text input
        house_id = st.selectbox("Choose a House", available_house_ids)
        house_index = X_test[X_test['house_id'] == house_id].index[0]
        actual_price = np.expm1(y_test.iloc[house_index].SalePrice).round(2)
        formatted_actual_price = f'{actual_price:,.2f}'
        st.write(f'Actual Price of House {house_id}: ${formatted_actual_price}')

        y_pred = model.predict(X_test.drop(['house_id'], axis=1))
        predicted_price = np.expm1(y_pred[house_index]).round(2) 
        formatted_predicted_price = f'{predicted_price:,.2f}'
        st.write(f"CatBoost Model's Price Prediction for House {house_id}: ${formatted_predicted_price}")

        difference = predicted_price - actual_price
        formatted_difference = f'{np.abs(difference):,.2f}'
        if difference > 0:
            st.write(f"Difference Between the Predicted and Actual Price: ${formatted_difference}")
        else:
            st.write(f"Difference Between the Predicted and Actual Price: -${formatted_difference}")
        plot_shap(model, data, house_id, X_train=X_train, X_test=X_test)

    # Call the conclusion and future work display function if the selected page is 'Conclusion and Future Work'
    elif page == 'Conclusion and Future Work':
        conclusion_future_work.display_conclusion_and_future_work()

    # Call the feature definitions display function if the selected page is 'Feature Definitions'
    elif page == 'Feature Definitions':
        feature_definitions.display_feature_definitions()
  
if __name__ == "__main__":
    main()