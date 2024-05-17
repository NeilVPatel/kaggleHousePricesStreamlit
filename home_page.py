import streamlit as st

def display_home():
    st.image("/mount/src/kagglehousepricesstreamlit/DALLE_image_for_homepage.webp")
    st.write("""
    The Kaggle House Prices competition provides a unique opportunity to apply advanced data analytics techniques to predict the final sale price of homes in Ames, Iowa. 
    In this project, I leverage a powerful CatBoost model to accurately estimate house prices based on a rich dataset encompassing various features of houses, such as size, neighborhood, year built, and overall quality.
    """)
    st.button("Competition Page", "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview")
    st.subheader("Navigation Guide")
    st.write("Use the sidebar to navigate through this application. Below are the available sections:")
    st.markdown(f"""
    - **User-based SHAP:** Select a specific house to view its actual vs. predicted price along with SHAP visualizations, offering detailed insights into how each feature influenced the prediction.
    - **Feature Importance:** Explore the features used to train the CatBoost model and understand their relative importance in predicting house prices.
    - **Exploratory Data Analysis (EDA):** Gain insights into the dataset through various visualizations and statistical analyses.
    - **Data Cleaning:** Understand the steps taken to preprocess and clean the dataset.
    - **Feature Engineering:** Learn about the new features created to improve the model's performance.
    - **Model Training and Evaluation:** Discover the process of training the CatBoost model and evaluating its performance.
    - **Conclusion and Future Work:** Review the project's key findings and potential future improvements.
    - **Feature Definitions:** Access detailed definitions of each feature in the dataset to aid in understanding and interpretation.
    """)