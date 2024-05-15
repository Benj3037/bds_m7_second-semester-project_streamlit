# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import altair as alt

# Import the functions from the features folder. This is the functions we have created to generate features for weather measures and calendar
from features import weather_measures, calendar 

# PART 2: Defining the functions for the Streamlit app
def print_fancy_header(text, font_width="bold", font_size=22, color="#2656a3"):
    res = f'<span style="font-width:{font_width}; color:{color}; font-size:{font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)  

def print_fancy_subheader(text, font_width="bold", font_size=22, color="#333"):
    res = f'<span style="font-width:{font_width}; color:{color}; font-size:{font_size}px;">{text}</span>'
    st.markdown(res, unsafe_allow_html=True)  

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
title = "📊 Electricity Price Prediction 🌦"

# Subtitle
st.markdown("""
            Welcome to the electricity price predicter for DK1. 
            \n The forecasted electricity prices are based on weather conditions, previous prices, and Danish holidays.
            Forecast prices are updated every 24 hours. 
            \nTaxes and fees are not included in the DKK prediction prices.
""")

st.write(3 * "-")

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
          
with st.expander("⚖️ **Model Performance**"):
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

# Display the predictions based on the user selection
st.write(3 * "-")
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Matrix for forecasted Electricity Prices", 
    "Linechart for forecasted Electricity Prices"]
)

filtered_predictions_df = predictions_df.head(date_range * 24)

# Matrix based on user selection
if visualization_option == "Matrix for forecasted Electricity Prices":
    # Prepare the data for the matrix
    data = filtered_predictions_df
    data['Date'] = data['time'].dt.strftime('%Y-%m-%d')
    data['Time of day'] = data['time'].dt.strftime('%H:%M')
    data.drop(columns=['time'], inplace=True)

    # Pivot the DataFrame
    pivot_df = data.pivot(index='Time of day', columns='Date', values='prediction')

    # Make a markdown description for the matrix
    st.markdown("""
            This is a matrix of the forecasted electricity prices for comming days. The user can change the date range in the sidebar.
            \n Each column represents a day and each row represents a time of day.
                
    """) 

    # Display the matrix
    st.write(pivot_df)  

# Linechart based on user selection
elif visualization_option == "Linechart for forecasted Electricity Prices":
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
