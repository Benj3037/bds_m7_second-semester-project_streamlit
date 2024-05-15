# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd


# PART 3: Page settings
st.set_page_config(
    page_title="About",
    page_icon="📚",
    layout="wide"
)

# Title for the streamlit app
st.title('📚 About')

# Subtitle
st.markdown("""
            Substitle. 
""")


# PART 3.1: Sidebar settings
with st.sidebar:
    
    st.write("© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen")


with st.expander("📕 **Data Engineering and Machine Learning Operations in Business**"):
                 st.markdown("""
Learning Objectives:
- Using our skills for designing, implementing, and managing data pipelines and ML systems.
- Focus on practical applications within a business context.
- Cover topics such as data ingestion, preprocessing, model deployment, monitoring, and maintenance.
- Emphasize industry best practices for effective operation of ML systems.
"""
)
                 
with st.expander("📝 **This assignment**"):
                st.markdown("""
The objective of this assignment is to build a prediction system that predicts the electricity prices in Denmark (area DK1) based on weather conditions, previous prices, and the Danish holidays.
"""
)