import streamlit as st

# Load feature definitions from a text file
def load_feature_definitions(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        definitions = {}
        feature = None
        for line in file:
            line = line.strip()
            if line and ':' in line:
                parts = line.split(":", 1)
                feature = parts[0].strip()
                definitions[feature] = parts[1].strip()
            elif line and feature:
                definitions[feature] += f"<br>{line.strip()}"
    return definitions

def display_feature_definitions():
    feature_definitions_path = '/mount/src/kagglehousepricesstreamlit/feature_definitions.txt'
    feature_definitions_dict = load_feature_definitions(feature_definitions_path)

    st.title('Feature Definitions')

    # Explanation of Feature Definitions Page
    st.subheader('Overview')
    st.markdown("""
    This page provides definitions and explanations for each feature in the dataset. Understanding these definitions can help in interpreting the results and making informed decisions based on the model's predictions.
    """)

    # Display each feature and its definition
    for feature, definition in feature_definitions_dict.items():
        st.markdown(f"### {feature}")
        st.markdown(definition)

    st.markdown("---")