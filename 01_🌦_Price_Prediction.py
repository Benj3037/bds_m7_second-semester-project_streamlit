# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import altair as alt
from st_pages import Page, show_pages, add_page_title

# Import the functions from the features folder. This is the functions we have created to generate features for weather measures and calendar
from features import electricity_prices, weather_measures, calendar

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
        name='dk1_electricity_training_feature_view',
        version=1
    )

    return feature_view

@st.cache_data()
def get_model():
    project = hopsworks.login()
    mr = project.get_model_registry()
    retrieved_model = mr.get_model(
        name="xgb_electricity_price_model",
        version=1
    )
    saved_model_dir = retrieved_model.download()
    retrieved_xgboost_model = joblib.load(saved_model_dir + "/xgb_electricity_price_model.pkl")

    return retrieved_xgboost_model

# Function to load the dataset
@st.cache_data()
def load_new_data():
    # Fetching weather forecast measures for the next 5 days
    weather_forecast_df = weather_measures.forecast_weather_measures(
        forecast_length=5
    )

    # Fetching danish calendar
    calendar_df = calendar.calendar_denmark(
        freq='H',
    )

    # Fetching the moving average of the electricity prices
    electricity_price_window_df = electricity_prices.electricity_prices_window(
        historical=False,
        area=["DK1"],
    )

    # Merging the weather forecast and electricity price window dataframes
    new_data = pd.merge(electricity_price_window_df, weather_forecast_df, how='inner', left_on='timestamp', right_on='timestamp')

    # Merging the new data and calendar dataframes
    new_data = pd.merge(new_data, calendar_df, how='inner', left_on='timestamp', right_on='timestamp')

    # Dropping and renaming columns for the new data with weather forecast and calendar
    new_data.drop(columns=['datetime_y', 'hour_y', 'date_y','datetime_x'], inplace=True)
    new_data.rename(columns={
        'date_x': 'date', 
        'hour_x': 'hour'}, inplace=True)

    return new_data

def load_predictions():
    # Drop columns 'date', 'datetime', 'timestamp' from the DataFrame 'new_data'
    data = load_new_data()[['hour', 'prev_1w_mean', 'prev_2w_mean', 'prev_4w_mean', 'prev_6w_mean',
       'prev_8w_mean', 'prev_12w_mean', 'temperature_2m',
       'relative_humidity_2m', 'precipitation', 'rain', 'snowfall',
       'weather_code', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m',
       'dayofweek', 'day', 'month', 'year', 'workday']]


    # Load the model and make predictions
    predictions = get_model().predict(data)

    # # Create a DataFrame with the predictions and the time
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

    if st.button("Clear Cache"):
        st.cache_data.clear()

    st.markdown("""<div style='text-align: center;'>
  <p>© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen</p>
</div>""", unsafe_allow_html=True)


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

with st.expander("📝 **Exam assignment**"):
    st.markdown(
        """
            This is the streamlit application for the final exam assigment. You will find the MLOps pipeline here: 
            \nGithub repository: [bds_m7_second-semester-project](https://github.com/tobiasmj97/bds_m7_second-semester-project)
        """
    )

with st.expander("🏋️ **Chosen model**"):
    st.markdown(
        """
            The model used in this application is an **XGBoost model** trained on historical data
        """
    )

# Display the predictions based on the user selection
st.write(3 * "-")

filtered_predictions_df = predictions_df.head(date_range * 24)

# Create two columns
col1, col2 = st.columns([4, 1])

# Place line chart in the first column
with col1:

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

# Place DataFrame representation in the second column
with col2:

    # Prepare the data for the matrix
    filtered_predictions_df['Date'] = filtered_predictions_df['time'].dt.strftime('%Y-%m-%d')
    filtered_predictions_df['Time of day'] = filtered_predictions_df['time'].dt.strftime('%H:%M')
    filtered_predictions_df.drop(columns=['time'], inplace=True)

    # Pivot the DataFrame
    pivot_df = filtered_predictions_df.pivot(index='Time of day', columns='Date', values='prediction')

    # Display the matrix
    st.write(pivot_df) 
