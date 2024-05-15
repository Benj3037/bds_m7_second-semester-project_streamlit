# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import altair as alt

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
    page_title="Electricity Price Prediction",
    page_icon="🌦",
    layout="wide"
)

# PART 3.1: Sidebar settings
with st.sidebar:
    
    # Sidebar progress bar
    progress_bar = st.sidebar.header('⚙️ Working Progress')
    progress_bar = st.sidebar.progress(0)

    login_hopswork()
    progress_bar.progress(40)

    get_model()
    progress_bar.progress(80)

    load_new_data()
    progress_bar.progress(100)

    # Sidebar filter: Date range
    predictions_df = load_predictions()

    min_value = 1
    max_value = int(len(predictions_df['time'].unique()) / 24)
    default = int(48 / 24)

    date_range = st.sidebar.slider("Select Date Range", min_value=min_value, max_value=max_value, value=default)

    st.write("© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen")

# PART 4: Main content

# Title for the streamlit app
st.title('Electricity Price Prediction 🌦')

# Subtitle
st.markdown("""
            Welcome to the electricity price predicter for DK1. 
            \n The forecasted electricity prices are based on weather conditions, previous prices, and Danish holidays.
            Forecast prices are updated every 24 hours. 
            \nTaxes and fees are not included in the DKK prediction prices.
""")

# Display the predictions based on the user selection
st.write(3 * "-")

filtered_predictions_df = predictions_df.head(date_range * 24)

# Linechart based on user selection
# Create Altair chart with line and dots
chart = alt.Chart(filtered_predictions_df).mark_line(point=True).encode(
    x='time:T',
    y='prediction:Q',
    tooltip=[alt.Tooltip('time:T', title='Date', format='%d-%m-%Y'), 
             alt.Tooltip('time:T', title='Time', format='%H:%M'), 
             alt.Tooltip('prediction:Q', title='Spot Price (DKK)', format='.2f')
            ]
)
# Make a markdown description for the line chart
st.markdown("""
        This is a line chart of the forecasted electricity prices for comming days. The user can change the date range in the sidebar.
        \n The plot is interactive which ables the user to hover over the line to see the exact price at a specific time.
""") 

# Display the chart
st.altair_chart(chart, use_container_width=True)
