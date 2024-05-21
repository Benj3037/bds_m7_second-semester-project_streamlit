## Performance metrics for the selected model

# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 

# Import the functions from the features folder. This is the functions we have created to generate features for weather measures and calendar
from features import weather_measures, calendar 

# PART 3: Page settings
st.set_page_config(
    page_title="Performance",
    page_icon="📊",
    layout="wide"
)

# Title for the streamlit app
st.title('📊 Performance')

# Subtitle
st.markdown("""
            Substitle. 
""")


# PART 3.1: Sidebar settings
with st.sidebar:
    
    st.write("© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen")



st.markdown("""
The model performance is evaluated using the following metrics:
- Mean Squared Error (MSE): The average of the squared differences between the predicted and actual values.
- R2 Score: The proportion of the variance in the dependent variable that is predictable from the independent variable.
- Mean Absolute Error (MAE): The average of the absolute differences between the predicted and actual values.

| Performance Metrics   | Value  |
|-----------------------|--------|
| MSE                   | 0.053  |
| R^2                   | 0.934  |
| MAE                   | 0.158  |
                   
""", unsafe_allow_html=True
)