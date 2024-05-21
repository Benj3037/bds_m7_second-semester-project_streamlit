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
def load_all_data():
    
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

    return data

def load_predictions_vs_actuals():

    data = load_all_data()

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

    # # Number of days to display
    # days_before = 5
    # last_indices = (24 * days_before)

    # # Display the predictions
    # predictions_df = predictions_df.iloc[:last_indices]

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
st.markdown(
    """
    Welcome to the Explore page for our Electricity Price Prediction project. This page is designed to help you delve into the data and gain valuable insights of the dataset. Here's what you can explore:

    - **Data Overview**: Get a summary of the dataset, including the number of records, range of dates, and key statistics on electricity prices.
    - **Visualizations**: Interactive charts and graphs such as time series plots, histograms, and box plots to analyze trends, seasonal patterns, and price distributions.
    - **Feature Analysis**: Examine correlations between electricity prices and various features like weather conditions, demand, and production sources.
    - **Filtering and Sorting**: Customize your view by filtering data by date ranges, regions, or other relevant criteria.
    - **Statistical Summaries**: Quick access to mean, median, standard deviation, and other statistical measures of electricity prices.

    Explore these features to uncover patterns and insights that will aid in accurately predicting electricity prices.
    """
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
    ### Data Overview:
    In this section, you'll find a concise summary of the dataset, including the total number of records, the range of dates covered, and essential statistics regarding electricity prices. Understanding the basic characteristics of the data is crucial for interpreting the visualizations and performing further analysis.
    Here we also provide a quick summaries of key statistical measures such as mean, median, and standard deviation for electricity prices. These summaries provide essential context and help identify outliers or unusual patterns in the data.
    """
)

# Total number of records
total_records = len(load_predictions_vs_actuals())
st.write(f"**Total number of records:** {total_records}")

# Range of dates
start_date = load_predictions_vs_actuals()['datetime'].min().strftime('%Y-%m-%d')
end_date = load_predictions_vs_actuals()['datetime'].max().strftime('%Y-%m-%d')
st.write(f"**Date range:** {start_date} to {end_date}")

# Key statistics on electricity prices
price_stats = load_predictions_vs_actuals()[['actuals', 'prediction']].describe()
st.write("**Key statistics on electricity prices:**")
st.write(price_stats)

st.write(3 * "-")
st.markdown(
    """
    ### Visualizations:
    Dive deeper into the data with a variety of interactive charts and graphs. From time series plots revealing trends over time to histograms and box plots showcasing price distributions, these visualizations provide invaluable insights into the patterns and fluctuations of electricity prices.
    """
)

# Display the predictions based on the user selection
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Matrix for forecasted Electricity Prices", 
    "Time Series Plot over the total time period",
    "Linechart for historical forcasted electricity prices vs actual electricity prices",
    "Box Plot of Electricity Prices",
    "Histogram of electricity prices"
    ]
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

    # Display the matrix
    st.write(pivot_df)  


elif visualization_option == "Time Series Plot over the total time period":
    # Create the Altair chart
    chart = alt.Chart(load_predictions_vs_actuals()).transform_fold(
        ['actuals', 'prediction'],
        as_=['Type', 'Value']
    ).mark_line().encode(
        x=alt.X('datetime:T', title='Date and Time'),
        y=alt.Y('Value:Q', title='Electricity Price (DKK)'),
        color=alt.Color('Type:N', title='Type'),
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

    # Display the chart
    st.altair_chart(chart, use_container_width=True)

# Linechart based on user selection
elif visualization_option == "Linechart for historical forcasted electricity prices vs actual electricity prices":
    
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
        y=alt.Y('Value:Q', title='Electricity Price (DKK)'),
        color=alt.Color('Type:N', title='Type'),
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
        y=alt.Y('Electricity Price:Q', title='Electricity Price (DKK)'),
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
    hist_actuals = alt.Chart(df_melted[df_melted['Type'] == 'actuals']).mark_bar(opacity=0.7).encode(
        alt.X('Electricity Price:Q', bin=alt.Bin(maxbins=50), title='Electricity Price'),
        alt.Y('count()', title='Count'),
        alt.Color('Type:N', legend=alt.Legend(title="Type", orient="top-right"))
    ).properties(
        width=600,
        height=400
    ).interactive()

    hist_predictions = alt.Chart(df_melted[df_melted['Type'] == 'prediction']).mark_bar(opacity=0.7, color='orange').encode(
        alt.X('Electricity Price:Q', bin=alt.Bin(maxbins=50), title='Electricity Price'),
        alt.Y('count()', title='Count'),
        alt.Color('Type:N', legend=alt.Legend(title="Type", orient="top-right"))
    ).properties(
        width=600,
        height=400
    ).interactive()

    hist_actuals_kde = alt.Chart(df_melted[df_melted['Type'] == 'actuals']).transform_density(
        density='Electricity Price',
        as_=['Electricity Price', 'density']
    ).mark_line(color='blue').encode(
        x='Electricity Price:Q',
        y='density:Q'
    )

    hist_predictions_kde = alt.Chart(df_melted[df_melted['Type'] == 'prediction']).transform_density(
        density='Electricity Price',
        as_=['Electricity Price', 'density']
    ).mark_line(color='orange').encode(
        x='Electricity Price:Q',
        y='density:Q'
    )

    histogram = alt.layer(hist_actuals, hist_actuals_kde, hist_predictions, hist_predictions_kde).resolve_scale(y='shared')

    st.altair_chart(histogram, use_container_width=True)

    # Histogram of electricity prices
    st.write("### Histogram of Electricity Prices")
    plt.figure(figsize=(10, 6))
    sns.histplot(load_predictions_vs_actuals()['actuals'], kde=True, color='blue', label='Actuals')
    sns.histplot(load_predictions_vs_actuals()['prediction'], kde=True, color='orange', label='Predictions')
    plt.xlabel('Electricity Price')
    plt.title('Distribution of Electricity Prices')
    plt.legend()
    st.pyplot(plt)

st.write(3 * "-")
st.markdown(
    """
    ### Feature Analysis:
    Investigate the relationship between electricity prices and various influencing factors such as weather conditions, demand fluctuations, and production sources. By exploring correlations and dependencies, you'll gain a deeper understanding of the dynamics driving price movements.
    """
)


# Drop 'datetime', 'date', and 'timestamp' columns
drop_for_corr = load_all_data().drop(columns=['timestamp','date','datetime'])

# Create the correlation matrix
correlation_matrix = drop_for_corr.corr()
 
# Set the size of the figure
plt.figure(figsize=(15, 12)) 
 
# Plot the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
 
# Display the plot
st.pyplot(plt)