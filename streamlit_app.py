# Import libraries
import shap
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyarrow import parquet as pq
from catboost import CatBoostRegressor, Pool
import joblib

# Path of the trained model and data
MODEL_PATH = "/mount/src/kagglehousepricesstreamlit/catboost_model.cbm"
DATA_PATH = "/mount/src/kagglehousepricesstreamlit/data.parquet"
IMAGE_PATH = "/mount/src/kagglehousepricesstreamlit/DALLE_image_for_homepage.webp"

# Set Page Title
st.set_page_config(page_title="House Prices Project")

# Define data, model, and train/test sets
@st.cache_data
def load_data():
    data = pd.read_parquet(DATA_PATH)
    return data

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
    plt.axhline(0, color='black', linewidth=0.5)
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
    plt.axhline(0, color='black', linewidth=0.5)
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
    
st.title("Kaggle House Prices Project")

def main():
    model = load_model()
    data = load_data()

    X_train = load_x_y("/mount/src/kagglehousepricesstreamlit/X_train.pkl")
    X_test = load_x_y("/mount/src/kagglehousepricesstreamlit/X_test.pkl")
    y_train = load_x_y("/mount/src/kagglehousepricesstreamlit/y_train.pkl")
    y_test = load_x_y("/mount/src/kagglehousepricesstreamlit/y_test.pkl")

    # Radio buttons for options in sidebar
    with st.sidebar:
        election = st.radio(
            "Make Your Choice:",
            ("Home", "User-based SHAP", "Feature Importance")
        )
    available_house_ids = X_test['house_id'].tolist()

    if election == "Home":
        st.image(IMAGE_PATH)
        st.write("The Kaggle House Prices competition provides a unique opportunity to apply advanced data analytics techniques to predict the final sale price of homes in Ames, Iowa. In this project, I leverage a powerful CatBoost model to accurately estimate house prices based on a rich dataset encompassing various features of houses, such as size, neighborhood, year built, and overall quality.")
        st.link_button("Competition Page", "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview")
        st.subheader("Use the sidebar to navigate through this application.") 
        st.markdown("**User-based SHAP:** Select a specific house to view its actual vs. predicted price along with SHAP visualizations, offering detailed insights into how each feature influenced the prediction.")
        st.markdown("**Feature Importance:** Explore the features used to train the CatBoost model and understand their relative importance in predicting house prices.")
        

    # If User-based SHAP option is selected
    if election == "User-based SHAP":
        # House ID text input
        house_id = st.selectbox("Choose a House", available_house_ids)
        house_index = X_test[X_test['house_id'] == house_id].index[0]
        st.write(f'Actual Price of House {house_id}: ${np.expm1(y_test.iloc[house_index].SalePrice).round(2)}')
        y_pred = model.predict(X_test.drop(['house_id'], axis=1))
        st.write(f"CatBoost Model's Price Prediction for House {house_id}: ${np.expm1(y_pred[house_id]).round(2)}")

        difference = np.expm1(y_pred[house_id]).round(2) - np.expm1(y_test.iloc[house_index].SalePrice)
        if difference > 0:
            st.write(f"Difference Between the Predicted and Actual Price: ${np.round(difference,2)}")
        else:
            st.write(f"Difference Between the Predicted and Actual Price: -${np.absolute(np.round(difference,2))}")
        plot_shap(model, data, house_id, X_train=X_train, X_test=X_test)

    # If Feature Importance is selected
    elif election == "Feature Importance":
        summary(model, data, X_train=X_train.drop('house_id',axis=1), X_test=X_test.drop('house_id',axis=1))

if __name__ == "__main__":
    main()