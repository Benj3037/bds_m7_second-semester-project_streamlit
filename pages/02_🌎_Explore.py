# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib

# Import the functions from the features folder. This is the functions we have created to generate features for weather measures and calendar
from features import weather_measures, calendar 

# PART 2: Defining the functions for the Streamlit app
# We want to cache several functions to avoid running them multiple times
@st.cache_data()
def login_hopswork():
    project = hopsworks.login()
    fs = project.get_feature_store()

    return fs

@st.cache_data()
def get_feature_view():
    project = hopsworks.login()
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(
        name='electricity_training_feature_view',
        version=1
    )

    return feature_view

@st.cache_data()
def get_model():
    project = hopsworks.login()
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(
        name="electricity_price_prediction_model",
        version=1
    )
    saved_model_dir = retrieved_model.download()
    retrieved_xgboost_model = joblib.load(saved_model_dir + "/dk_electricity_model.pkl")

    return retrieved_xgboost_model

# Function to load the dataset
def load_new_data():
    # Fetching weather forecast measures for the next 5 days
    weather_forecast_df = weather_measures.forecast_weather_measures(
        forecast_length=5
    )

    # Fetching danish calendar
    calendar_df = calendar.dk_calendar()

    # Merging the weather forecast and calendar dataframes
    new_data = pd.merge(weather_forecast_df, calendar_df, how='inner', left_on='date', right_on='date')

    return new_data

def load_predictions():
    # Drop columns 'date', 'datetime', 'timestamp' from the DataFrame 'new_data'
    data = load_new_data().drop(columns=['date', 'datetime', 'timestamp'])

    # Load the model and make predictions
    predictions = get_model().predict(data)

    # Create a DataFrame with the predictions and the time
    predictions_data = {
        'prediction': predictions,
        'time': load_new_data()["datetime"],
    }

    predictions_df = pd.DataFrame(predictions_data).sort_values(by='time')

    return predictions_df

# PART 3: Page settings
st.set_page_config(
    page_title="Explore",
    page_icon="🌎",
    layout="wide"
)

# Title for the streamlit app
st.title('🌎 Explore')

# Subtitle
st.markdown("""
            Substitle. 
""")

# PART 3.1: Sidebar settings
with st.sidebar:
    
    st.write("© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen")

# Make a markdown description for the matrix
st.markdown("""
        This is a matrix of the forecasted electricity prices for comming days. The user can change the date range in the sidebar.
        \n Each column represents a day and each row represents a time of day.
""") 

# Prepare the data for the matrix
data = load_predictions()
data['Date'] = data['time'].dt.strftime('%Y-%m-%d')
data['Time of day'] = data['time'].dt.strftime('%H:%M')
data.drop(columns=['time'], inplace=True)

# Pivot the DataFrame
pivot_df = data.pivot(index='Time of day', columns='Date', values='prediction')

# Display the matrix
st.write(pivot_df) 



