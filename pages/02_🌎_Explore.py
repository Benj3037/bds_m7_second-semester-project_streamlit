# PART 1: Importing the necessary libraries
import streamlit as st
import pandas as pd
import hopsworks 
import joblib
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

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
    forecast_weather_df = weather_measures.forecast_weather_measures(
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
    new_data = pd.merge(electricity_price_window_df, forecast_weather_df, how='inner', left_on='timestamp', right_on='timestamp')

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

@st.cache_data()
def load_all_data():
    
    # Fetching historical electricity prices for area DK1 from January 1, 2022
    electricity_price_df = electricity_prices.electricity_prices(
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
    forecast_weather_df = weather_measures.forecast_weather_measures(
        forecast_length=5
    )

    # Fetching danish calendar
    danish_calendar_df = calendar.calendar_denmark(
        freq='D',
        start='2020-01-01', 
        end=pd.Timestamp(pd.Timestamp.now().year + 1, 12, 31).date()
    )

    # Fetching electricity_price_window_df
    electricity_price_window_df = electricity_prices.electricity_prices_window(
        historical=True,
        area=["DK1"],
        start='2020-01-01'
    )

    # Concatenate the historical weather and weather forecast DataFrames
    weather = pd.concat([historical_weather_df, forecast_weather_df])

    # Merging the weather forecast and calendar dataframes
    data = pd.merge(electricity_price_df, danish_calendar_df, how='inner', left_on='date', right_on='date')
    
    # Dropping and renameing columns
    data.drop(columns=['datetime_y', 'hour_y', 'timestamp_y'], inplace=True)
    data.rename(columns={
        'timestamp_x': 'timestamp', 
        'datetime_x': 'datetime', 
        'hour_x': 'hour'
    }, inplace=True)

    # Merging the electricity_df and calendar_df with the historical weather data
    data = pd.merge(data, historical_weather_df, how='inner', left_on='timestamp', right_on='timestamp')
    # Dropping and renameing columns
    data.drop(columns=['datetime_y', 'date_y', 'hour_y'], inplace=True)
    data.rename(columns={
        'datetime_x': 'datetime', 
        'date_x': 'date', 
        'hour_x': 'hour', 
    }, inplace=True)
    
    # Merging the electricity_df and calendar_df and historical_weather_df with the electricity window 
    data = pd.merge(data, electricity_price_window_df, how='inner', left_on='timestamp', right_on='timestamp')

    # Dropping and renameing columns
    data.drop(columns=['datetime_y'], inplace=True)
    data.rename(columns={
        'datetime_x': 'datetime', 
    }, inplace=True)

    data = data.sort_values(by='datetime', ascending=False) 

    return data

def load_predictions_vs_actuals():

    data = load_all_data()

    # Drop columns 'date', 'datetime', 'timestamp' from the DataFrame 'data'
    X2 = data.drop(columns=['date', 'datetime', 'timestamp'])
    X = X2[['hour', 'prev_1w_mean', 'prev_2w_mean', 'prev_4w_mean', 'prev_6w_mean',
       'prev_8w_mean', 'prev_12w_mean', 'temperature_2m',
       'relative_humidity_2m', 'precipitation', 'rain', 'snowfall',
       'weather_code', 'cloud_cover', 'wind_speed_10m', 'wind_gusts_10m',
       'dayofweek', 'day', 'month', 'year', 'workday']]
    y = X2.pop('dk1_spotpricedkk_kwh')

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

    return predictions_df

# PART 3: Page settings
st.set_page_config(
    page_title="Explore",
    page_icon="🌎",
    layout="wide"
)

# Title for the streamlit app
st.title('🌎 Explore')

# Description of the Explore page
st.markdown(
    """
    Welcome to the Explore page for our Electricity Price Prediction project. This page is designed to help you delve into the data and gain valuable insights from the dataset. Here's what you can explore:

    - <a href="#data-overview" style="text-decoration: underline; color: inherit;">**Data Overview**</a>: Get a summary of the dataset, including the number of records, range of dates, and key statistics on electricity prices.
    - <a href="#visualizations" style="text-decoration: underline; color: inherit;">**Visualizations**</a>: Interactive charts and graphs such as time series plots, histograms, and box plots to analyze trends, seasonal patterns, and price distributions.
    - <a href="#feature-analysis" style="text-decoration: underline; color: inherit;">**Feature Analysis**</a>: Examine correlations between electricity prices and various features like weather conditions, demand, and production sources.

    Explore these features to uncover patterns and insights that will aid in accurately predicting electricity prices.
    """, unsafe_allow_html=True
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

    load_predictions_vs_actuals()
    progress_bar.progress(100)

    if st.button("Clear Cache"):
        st.cache_data.clear()
        # load_predictions_vs_actuals()

    st.markdown("""<div style='text-align: center;'>
  <p>© 2024 Camilla Dyg Hannesbo, Benjamin Ly, Tobias Moesgård Jensen</p>
</div>""", unsafe_allow_html=True)

st.write(3 * "-")
st.markdown(
    """
    ### Data Overview
    In this section, you'll find a concise summary of the dataset, including the total number of records, the range of dates covered, and essential statistics regarding electricity prices. Understanding the basic characteristics of the data is crucial for interpreting the visualizations and performing further analysis.
    Here we also provide a quick summaries of key statistical measures such as mean, median, and standard deviation for electricity prices. These summaries provide essential context and help identify outliers or unusual patterns in the data.
    """
)

# Range of dates
start_date = load_predictions_vs_actuals()['datetime'].min().strftime('%Y-%m-%d')
end_date = load_predictions_vs_actuals()['datetime'].max().strftime('%Y-%m-%d')
st.write(f"**Date range:** {start_date} to {end_date}")

# Key statistics on electricity prices
price_stats = load_predictions_vs_actuals()[['actuals', 'prediction']].describe()
st.write("**Key statistics on electricity prices:**")
st.write(price_stats.T)

st.write(3 * "-")
st.markdown(
    """
    ### Visualizations
    Dive deeper into the data with a variety of interactive charts and graphs. From time series plots revealing trends over time to histograms and box plots showcasing price distributions, these visualizations provide invaluable insights into the patterns and fluctuations of electricity prices.
    """
)

# Display the predictions based on the user selection
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Linechart for historical predicted electricity prices vs actual electricity prices",
    "Box Plot of Electricity Prices",
    "Histogram of electricity prices"
    ]
)

# Linechart based on user selection
if visualization_option == "Linechart for historical predicted electricity prices vs actual electricity prices":
    
    min_value = 1
    max_value = 5
    default = int(48 / 24)

    date_range = st.slider("Select Date Range", min_value=min_value, max_value=max_value, value=default)
    filtered_predictions_df = load_predictions_vs_actuals().head(date_range * 24)
    
    # Create the Altair chart
    chart = alt.Chart(filtered_predictions_df).transform_fold(
        ['actuals', 'prediction'],
        as_=['Type', 'Value']
    ).mark_line(point=True).encode(
        x=alt.X('datetime:T', title='Date and Time'),
        y=alt.Y('Value:Q', title='Electricity Price (DKK/kWh)'),
        color=alt.Color('Type:N', title='Type'),
        tooltip=[alt.Tooltip('datetime:T', title='Date', format='%Y-%m-%d'), 
                alt.Tooltip('datetime:T', title='Time', format='%H:%M'), 
                alt.Tooltip('Value:Q', title='Actuals (DKK/kWh)', format='.4f'),
                alt.Tooltip('prediction:Q', title='Prediction (DKK/kWh)', format='.4f')
                ]
    ).properties(
        title='Actual vs Prediction over time',
        width=600,
        height=400
    ).interactive()  # Make the chart interactive

    # Make a markdown description for the matrix
    st.markdown("""
            This is a linechart of the forecasted electricity prices compared with the actual electricity prices. The user can change the date range above to show the last X dates.
                
    """) 

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

elif visualization_option == "Box Plot of Electricity Prices":

    # Transform the data for Altair
    df_melted = load_predictions_vs_actuals().melt(id_vars=['datetime'], value_vars=['actuals', 'prediction'], var_name='Type', value_name='Electricity Price')

    # Create the box plot using Altair
    box_plot = alt.Chart(df_melted).mark_boxplot(size=250).encode(
        x=alt.X('Type:N', title='Type'),
        y=alt.Y('Electricity Price:Q', title='Electricity Price (DKK/kWh)'),
        color=alt.Color('Type:N', title='Type'),
    ).properties(
        title='Box Plot of Actuals and Predictions',
        width=600,
        height=400
    )

    st.altair_chart(box_plot, use_container_width=True)

elif visualization_option == "Histogram of electricity prices":

    # Transform the data for Altair
    df_melted = load_predictions_vs_actuals().melt(id_vars=['datetime'], value_vars=['actuals', 'prediction'], var_name='Type', value_name='Electricity Price')

    # Create the histogram using Altair
    hist_actuals = alt.Chart(df_melted[df_melted['Type'] == 'actuals']).mark_bar(opacity=0.7, color='blue').encode(
        alt.X('Electricity Price:Q', bin=alt.Bin(maxbins=50), title='Electricity Price (DKK/kWh)'),
        alt.Y('count()', title='Count'),
        alt.Color('Type:N', legend=alt.Legend(title="Type", orient="top-right"))
    ).properties(
        width=600,
        height=400
    ).interactive()

    hist_predictions = alt.Chart(df_melted[df_melted['Type'] == 'prediction']).mark_bar(opacity=0.7, color='orange').encode(
        alt.X('Electricity Price:Q', bin=alt.Bin(maxbins=50), title='Electricity Price (DKK/kWh)'),
        alt.Y('count()', title='Count', axis=None),
        alt.Color('Type:N', legend=alt.Legend(title="Type", orient="top-right"))
    ).properties(
        width=600,
        height=400
    ).interactive()

    # Create the KDE overlays using Altair
    hist_actuals_kde = alt.Chart(df_melted[df_melted['Type'] == 'actuals']).transform_density(
        density='Electricity Price',
        as_=['Electricity Price', 'density']
    ).mark_line(color='blue').encode(
        x='Electricity Price:Q',
        y=alt.Y('density:Q', axis=None),
        tooltip=[alt.Tooltip('density:Q', title='Density')]
    )

    hist_predictions_kde = alt.Chart(df_melted[df_melted['Type'] == 'prediction']).transform_density(
        density='Electricity Price',
        as_=['Electricity Price', 'density']
    ).mark_line(color='orange').encode(
        x='Electricity Price:Q',
        y=alt.Y('density:Q', axis=None),
        tooltip=[alt.Tooltip('density:Q', title='Density')]
    )

    # Combine the histograms and KDE overlays, with separate y-axes
    histogram = alt.layer(
        hist_actuals, hist_predictions
    ).resolve_scale(
        y='independent'
    ).properties(
        width=600,
        height=400
    )

    kde_overlay = alt.layer(
        hist_actuals_kde, hist_predictions_kde
    ).resolve_scale(
        y='independent'
    ).properties(
        width=600,
        height=400
    )

    # Combine the histogram and KDE overlay with dual axes
    combined_chart = alt.layer(histogram, kde_overlay).resolve_scale(
        y='independent'
    ).properties(
        title='Actual vs Prediction Electricity Prices with KDE',
        width=600,
        height=400
    )

    # Display the combined chart in Streamlit
    st.altair_chart(combined_chart, use_container_width=True)

    # # Histogram of electricity prices
    # st.write("### Histogram of Electricity Prices")
    # plt.figure(figsize=(10, 6))
    # sns.histplot(load_predictions_vs_actuals()['actuals'], kde=True, color='blue', label='Actuals')
    # sns.histplot(load_predictions_vs_actuals()['prediction'], kde=True, color='orange', label='Predictions')
    # plt.xlabel('Electricity Price')
    # plt.title('Distribution of Electricity Prices')
    # plt.legend()
    # st.pyplot(plt)

st.write(3 * "-")
# st.subheader("Feature Analysis")
st.markdown(
    """
    ### Feature Analysis
    Investigate the relationship between electricity prices and various influencing factors such as weather conditions, demand fluctuations, and production sources. By exploring correlations and dependencies, you'll gain a deeper understanding of the dynamics driving price movements.
    """
)


# Drop 'datetime', 'date', and 'timestamp' columns
drop_for_corr = load_all_data().drop(columns=[ 'timestamp', 
                                           'date',
                                           'datetime', 
                                           'prev_1w_mean', 
                                           'prev_2w_mean', 
                                           'prev_4w_mean', 
                                           'prev_6w_mean', 
                                           'prev_8w_mean', 
                                           'prev_12w_mean', 
                                           ]
)

reorder_drop_for_corr = drop_for_corr[['dk1_spotpricedkk_kwh', 'hour', 'dayofweek', 'day', 'month', 'year',
       'workday', 'temperature_2m', 'relative_humidity_2m', 'precipitation',
       'rain', 'snowfall', 'weather_code', 'cloud_cover', 'wind_speed_10m',
       'wind_gusts_10m']]

# Create the correlation matrix
correlation_matrix = reorder_drop_for_corr.corr()
 
# Set the size of the figure
plt.figure(figsize=(15, 12)) 
 
# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
 
# Display the plot
st.pyplot(plt)