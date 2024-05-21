# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import altair as alt

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

@st.cache_data()
def load_predictions_vs_actuals():
    
    # Fetching historical electricity prices for area DK1 from January 1, 2022
    electricity_df = electricity_prices.electricity_prices(
        historical=True, 
        area=["DK1"], 
        start='2022-01-01'
    )

    # Fetching historical weather measurements from January 1, 2022
    historical_weather_df = weather_measures.historical_weather_measures(
        historical=True, 
        start = '2022-01-01'
    )

    # Fetching weather forecast measures for the next 5 days
    weather_forecast_df = weather_measures.forecast_weather_measures(
        forecast_length=5
    )

    # Fetching danish calendar
    calendar_df = calendar.dk_calendar()

    # Concatenate the historical weather and weather forecast DataFrames
    weather = pd.concat([historical_weather_df, weather_forecast_df])

    # Merging the weather forecast and calendar dataframes
    data = pd.merge(electricity_df, weather, how='right', left_on='timestamp', right_on='timestamp')
    data = data.drop(columns=['datetime_x', 'date_x', 'hour_x'])
    data = data.rename(columns={'datetime_y': 'datetime', 'date_y': 'date', 'hour_y': 'hour'})

    data = pd.merge(data, calendar_df, how='inner', left_on='date', right_on='date')
    data = data.sort_values(by='datetime', ascending=False) 

    # Drop columns 'date', 'datetime', 'timestamp' from the DataFrame 'data'
    y = data['dk1_spotpricedkk_kwh']
    X = data.drop(columns=['dk1_spotpricedkk_kwh', 'date', 'datetime', 'timestamp'])

    # Load the model and make predictions
    predictions = get_model().predict(X)

    # Create a DataFrame with the predictions and the time
    predictions_data = {
        'prediction': predictions,
        'actuals': y,
        'datetime': pd.to_datetime(X[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H'),
    }

    predictions_df = pd.DataFrame(predictions_data)#.sort_values(by='datetime', ascending=False)
    predictions_df.dropna(inplace=True)

    # Number of days to display
    days_before = 5
    last_indices = (24 * days_before)

    # Display the predictions
    predictions_df = predictions_df.iloc[:last_indices]

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
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        # load_predictions_vs_actuals()

    st.write("© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen")

# Display the predictions based on the user selection
st.write(3 * "-")
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Matrix for forecasted Electricity Prices", 
    "Linechart for forecasted Electricity Prices"]
)

# Matrix based on user selection
if visualization_option == "Matrix for forecasted Electricity Prices":
    # Prepare the data for the matrix
    data = load_predictions()
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
    
    min_value = 1
    max_value = int(len(load_predictions_vs_actuals()['datetime'].unique()) / 24)
    default = int(48 / 24)

    date_range = st.slider("Select Date Range", min_value=min_value, max_value=max_value, value=default)
    filtered_predictions_df = load_predictions_vs_actuals().head(date_range * 24)
    
    # Create the Altair chart
    chart = alt.Chart(filtered_predictions_df).transform_fold(
        ['actuals', 'prediction'],
        as_=['Type', 'Value']
    ).mark_line(point=True).encode(
        x='datetime:T',
        y='Value:Q',
        color='Type:N',
        tooltip=[alt.Tooltip('datetime:T', title='Date', format='%Y-%m-%d'), 
                alt.Tooltip('datetime:T', title='Time', format='%H:%M'), 
                alt.Tooltip('Value:Q', title='Actuals (DKK)', format='.4f'),
                alt.Tooltip('prediction:Q', title='Prediction (DKK)', format='.4f')
                ]
    ).properties(
        title='Actual vs Prediction over time',
        width=600,
        height=400
    ).interactive()  # Make the chart interactive

    # Make a markdown description for the matrix
    st.markdown("""
            This is a linechart of the forecasted electricity prices for comming days. The user can change the date range in the sidebar.
            \n Each column represents a day and each row represents a time of day.
                
    """) 

    # Display the chart
    st.altair_chart(chart, use_container_width=True)